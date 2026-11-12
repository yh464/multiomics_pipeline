#!/usr/bin/env python3
'''
Author: Yuankai He
Correspondence: yh464@cam.ac.uk
Version 1: 2026-11-04

Pipelines to identify gene programmes from scATAC-seq data
'''

# scanpy takes long to import so needs to be imported in each sub-function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os, time, pickle
from fnmatch import fnmatch
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from _utils.adatatools import subset_h5ad, downsample_adata

from _plots.corr_heatmap import corr_heatmap, corr_heatmap_wide_format
from _plots.colourcode_scatterplot import scatterplot_noaxis, scatterplot_adata
from _plots.regplot import temporal_regplot
sns.set_theme(style = 'ticks', font_scale = 1.2, palette = 'colorblind')
mpl.rcParams['font.family'] = 'Arial'

from _utils.logger import logger
log = logger()
from _utils.path import project
proj = project()

# region: cisTopic pipeline
def check_cistopic_complete(dataset, prefix, n_components = range(25, 501, 25), niter = 100):
    outdir = os.path.dirname(proj.to_pathname('atac_programmes_cistopic', dataset, prefix))
    os.makedirs(f'{outdir}/temp', exist_ok = True)
    incomplete_idx = []
    for i, n_topics in enumerate(n_components):
        if not os.path.isfile(f'{outdir}/temp/{prefix}.cistopic.k_{n_topics}.{niter}iter.model.pkl'):
            incomplete_idx.append(i)
    return incomplete_idx

def run_cistopic(dataset, prefix, outdir, n_components = range(25, 501, 25), worker = 0, niter = 100, 
    alpha = 50, eta = 0.1, mallet = '/rds/project/rds-Nl99R8pHODQ/toolbox/mallet', downsample = -1,
    cell_type = ['cell_type'], embedding = ['X_umap', 'X_tsne'], time_key = ['pseudotime', 'age', 'time'],
    seed = 19260817, force = False, savedir = None):

    import scanpy as sc
    from scipy import sparse
    import pickle
    from pycisTopic.cistopic_class import create_cistopic_object
    from pycisTopic.lda_models import run_cgs_models_mallet
    
    n_topics = n_components[worker]
    if len(n_components) == 1 and niter < 200:
        log.warn(f'n_iter < 200 is usually suitable only for model screening, consider increasing n_iter for final model fitting', calling_file = 'run_cistopic')
    os.makedirs(f'{outdir}/temp', exist_ok = True)

    # downsample cells if requested
    adata = sc.read_h5ad(proj.to_pathname('raw', dataset, prefix), 'r')
    adata = downsample_adata(adata, group_by = cell_type[0] if len(cell_type) > 0 else None, downsample = downsample, seed = seed)

    if adata.shape[0] > 150000:
        log.warn(f'Number of cells in {dataset} {prefix} is {adata.shape[0]}. This may take a long time to run and may require more memory. Consider downsampling.', calling_file = 'run_cistopic')
    X = sparse.csr_matrix(adata.X.T)
    cell_names = adata.obs_names.tolist()
    peak_names = adata.var_names.tolist()
    cistopic_obj = create_cistopic_object(X, cell_names, peak_names)
    cistopic_obj.add_cell_data(adata.obs)

    # run cistopic model
    if not os.path.isfile(f'{outdir}/temp/{prefix}.cistopic.k_{n_topics}.{niter}iter.model.pkl') or force:
        log.log(f'Running cistopic model for {dataset} {prefix} with {n_topics} topics', calling_file = 'run_cistopic')

        from _utils.slurm import get_slurm_config
        n_cpu, mem = get_slurm_config() # memory in MB
        os.environ['MALLET_MEMORY'] = f'{mem*0.9/1024:.0f}G'
        
        models = run_cgs_models_mallet(cistopic_obj, n_topics = [n_topics], n_iter = niter, n_cpu = n_cpu,
            alpha = alpha, eta = eta, alpha_by_topic = True, eta_by_topic = False,
            random_state = seed, save_path = f'{outdir}/temp', mallet_path = mallet,
            reuse_corpus = True, tmp_path = f'{outdir}/temp'
            )
        with open(f'{outdir}/temp/{prefix}.cistopic.k_{n_topics}.{niter}iter.model.pkl', 'wb') as f: pickle.dump(models[0], f)

    if len(check_cistopic_complete(dataset, prefix, n_components = n_components)) > 0:
        log.log('Waiting for other workers to finish cistopic model fitting', calling_file = 'run_cistopic')
        return

    # if all workers have finished, clean cache files that cannot be reused
    os.system(f'rm {outdir}/temp/*state.mallet.gz')
    os.system(f'rm {outdir}/temp/*inferencer.mallet')
    os.system(f'rm {outdir}/temp/*doctopics.txt')
    os.system(f'rm {outdir}/temp/*topickeys.txt')

    # compare models
    if len(n_components) > 1:
        log.log(f'Comparing cistopic models for {dataset} {prefix}', calling_file = 'run_cistopic')
        model_list = []
        for n_topics in n_components:
            with open(f'{outdir}/temp/{prefix}.cistopic.k_{n_topics}.{niter}iter.model.pkl', 'rb') as f: model_list.append(pickle.load(f))

        k_selection_metrics = pd.concat([m.metrics.assign(k = m.n_topic) for m in model_list]).set_index('k').sort_index().reset_index()
        k_selection_metrics.columns = ['k','arun_2010', 'cao_juan_2009', 'mimno_2011', 'loglikelihood']
        k_selection_metrics.to_csv(f'{outdir}/{prefix}.cistopic.k_selection_metrics.txt', index = False, sep = '\t')
        k_selection_metrics[['arun_2010', 'cao_juan_2009']] *= -1

        # rescale all metrics to [0, 1] for comparison
        k_selection_metrics_scaled = k_selection_metrics.copy()
        for col in ['arun_2010', 'cao_juan_2009', 'mimno_2011', 'loglikelihood']:
            k_selection_metrics_scaled[col] = (k_selection_metrics[col] - k_selection_metrics[col].min()) / (k_selection_metrics[col].max() - k_selection_metrics[col].min())
        
        # region: plot scaled metrics, similar to gene_programmes.py lines 423-447
        fig, ax0 = plt.subplots(figsize = (5,2))
        ax1 = ax0.twinx(); ax2 = ax0.twinx(); ax3 = ax0.twinx()
        for plot_ax, stat, label, colour, style in zip(
            [ax0, ax1, ax2, ax3],
            ['arun_2010', 'cao_juan_2009', 'mimno_2011', 'loglikelihood'],
            ['Arun (2010)', 'Cao (2009)', 'Mimno (2011)', 'Log-likelihood'],
            ['#1f78b4','#33a02c', '#6a3d9a','#000000'],
            ['--', '-', '-.', ':']
        ):
            plot_ax.plot(k_selection_metrics.k, k_selection_metrics[stat], label = label, c = colour, linestyle = style, marker = 'none', linewidth = 2)
            plot_ax.spines[['top', 'right', 'bottom']].set_visible(False)
            plot_ax.set_xticks(np.arange(min(n_components), max(n_components)+1, 5))
            plot_ax.tick_params(axis = 'x', bottom = True, labelbottom = True)
            plot_ax.yaxis.set_label_position('left')
            plot_ax.yaxis.tick_left()
            plot_ax.spines['left'].set_color(colour)
            plot_ax.tick_params(axis = 'y', colors = colour)
            plot_ax.set_ylabel('')
        ax0.grid(True, 'both','x')
        ax1.spines['left'].set_position(('outward', 40))
        ax2.spines['left'].set_position(('outward', 80))
        ax3.spines['left'].set_position(('outward', 120))
        ax0.set_xlabel('Number of components (k)', fontsize = 14)
        lines = ax0.get_lines() + ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
        ax1.legend(lines[:4], ['Arun (2010) inverted', 'Cao (2009) inverted', 'Mimno (2011)', 'Log-likelihood'], loc = 'upper center', frameon = False,
            bbox_to_anchor = (0.3, -0.3), ncol = 2, columnspacing = 0.5, fontsize = 12)
        fig.savefig(f'{outdir}/{prefix}.cistopic.k_selection_metrics.pdf', bbox_inches = 'tight')
        # endregion
            
        # compare using just mimno and loglikelihood
        k_selection_metrics_scaled['score'] = k_selection_metrics_scaled[['mimno_2011', 'loglikelihood']].mean(axis = 1)
        k_optim = k_selection_metrics_scaled.loc[k_selection_metrics_scaled['score'].idxmax(), 'k']
        log.log(f'Automatically selecting {k_optim} topics for {dataset} {prefix} based on mimno and loglikelihood', calling_file = 'run_cistopic')
        log.warn(f'Please check {outdir}/{prefix}.cistopic.k_selection_metrics.pdf for details', calling_file = 'run_cistopic')
    else: 
        k_optim = n_components[0]

    # export best model loadings and binarised sets
    with open(f'{outdir}/temp/{prefix}.cistopic.k_{k_optim}.{niter}iter.model.pkl', 'wb') as f:
        best_model = pickle.load(f)
    best_model.topic_region.T.to_csv(f'{outdir}/{prefix}.cistopic.k_{k_optim}.topic_loadings.txt.gz', sep = '\t')
    cell_topic = best_model.cell_topic.T # rows are now cells, columns are topics
    # normalise such that each cell sums to one
    cell_topic = cell_topic.div(cell_topic.sum(axis = 1), axis = 0)
    cell_topic.to_parquet(f'{outdir}/{prefix}.cistopic.k_{k_optim}.cell_topic.parquet', compression = 'gzip')

    cistopic_obj.add_LDA_model(best_model)

    from pycisTopic.topic_binarization import binarize_topics
    binarized_topics = binarize_topics(cistopic_obj, method = 'otsu')
    with open(f'{outdir}/{prefix}.cistopic.k_{k_optim}.binarised_topics.txt', 'w') as f: # MAGMA format
        for k, peaks in binarized_topics.items():
            print(' '.join([k] + list(peaks)), file = f)

    from pycisTopic.utils import region_names_to_coordinates
    binarised_topics_bed = {k: region_names_to_coordinates(peaks).sort_values(by = ['Chromosome','Start','End']) for k, peaks in binarized_topics.items()}

    # TODO greatpy
# endregion

# region: scOpen pipeline
def check_scopen_complete(dataset, prefix, n_components = range(25, 501, 25)):
    outdir = os.path.dirname(proj.to_pathname('atac_programmes_scopen', dataset, prefix))
    os.makedirs(f'{outdir}/temp', exist_ok = True)
    incomplete_idx = []
    for i, n_topics in enumerate(n_components):
        if not os.path.isfile(f'{outdir}/temp/{prefix}.scopen.k_{n_topics}.weights.txt'):
            incomplete_idx.append(i)
    return incomplete_idx

def run_scopen(dataset, prefix, outdir, n_components = range(25, 501, 25), n_peaks = 100000, worker = 0,
    downsample = -1, cell_type = ['cell_type'], embedding = ['X_umap', 'X_tsne'], time_key = ['pseudotime', 'age', 'time'],
    seed = 19260817, force = False, savedir = None):

    import scanpy as sc
    import anndata as ad
    from kneed import KneeLocator
    from scopen.Main import tf_idf_transform, run_nmf
    from _utils.gadgets import mv_symlink

    os.makedirs(f'{outdir}/temp', exist_ok = True)

    # downsample cells if requested
    temp_h5ad_path = f'{outdir}/temp/{prefix}.hvp.h5ad'
    tfidf_path = f'{outdir}/temp/{prefix}.tfidf.h5ad'
    if worker == 0 and (not os.path.isfile(tfidf_path) or force):
        adata = sc.read_h5ad(proj.to_pathname('raw', dataset, prefix), 'r')
        adata = downsample_adata(adata, group_by = cell_type[0] if len(cell_type) > 0 else None, downsample = downsample, seed = seed)
        adata.X.data[:] = 1 # convert to binary matrix
        adata.X = adata.X.astype('int8')
        sc.pp.filter_cells(adata, min_counts = 1000)
        sc.pp.filter_genes(adata, min_cells = 10)

        # select highly variable peaks
        sc.pp.highly_variable_genes(adata, n_top_genes = n_peaks, flavor = 'seurat_v3')
        adata_hvp = adata[:, adata.var['highly_variable']].copy()
        sc.write(temp_h5ad_path, adata_hvp)
        del adata

        # run tf-idf transformation
        adata_tfidf = ad.AnnData(X = tf_idf_transform(adata_hvp.X.T).T, obs = adata_hvp.obs.copy(), var = adata_hvp.var.copy())
        sc.write(tfidf_path, adata_tfidf)
        del adata_hvp

    elif os.path.isfile(tfidf_path):
        adata_tfidf = sc.read_h5ad(tfidf_path)
    else:
        log.warn('TF-IDF transformation not complete, please run again after worker 0 completes.', calling_file = 'run_scopen')
        return

    # run NMF decomposition based on TF-IDF transformed matrix
    k = n_components[worker]
    if not os.path.isfile(f'{outdir}/temp/{prefix}.scopen.k_{k}.weights.txt') or force:
        log.log(f'Running scOpen model for {dataset} {prefix} with {k} components', calling_file = 'run_scopen')
        nmf_output = run_nmf((adata_tfidf.X.T, k, 1, 200, 1, seed, 'nndsvd')) # alpha = 1, 200 iters, verbose = 1
        W = nmf_output[0]; H = nmf_output[1]; error = nmf_output[2]
        np.savetxt(f'{outdir}/temp/{prefix}.scopen.k_{k}.weights.txt', W, delimiter = '\t')
        np.savetxt(f'{outdir}/temp/{prefix}.scopen.k_{k}.scores.txt', H, delimiter = '\t')
        with open(f'{outdir}/temp/{prefix}.scopen.k_{k}.error.txt', 'w') as f: print(error, file = f)
    log.log(f'scOpen model for {dataset} {prefix} with {k} components completed', calling_file = 'run_scopen')
    log.log(f'Output saved to {outdir}/temp/{prefix}.scopen.k_{k}.weights.txt', calling_file = 'run_scopen')

    # compare all values of k
    if len(check_scopen_complete(dataset, prefix, n_components = n_components)) > 0:
        log.log('Waiting for other workers to finish scOpen model fitting', calling_file = 'run_scopen')
        return

    df = pd.DataFrame(dict(n_components = n_components, 
        error = [float(open(f'{outdir}/temp/{prefix}.scopen.k_{k}.error.txt').read().strip()) for k in n_components], knee = False))
    kl = KneeLocator(df.n_components, df.error, curve = 'convex', direction = 'decreasing')
    df.loc[df.n_components == kl.knee, 'knee'] = True
    df.to_csv(f'{outdir}/{prefix}.scopen.k_selection_metrics.txt', index = False, sep = '\t')
    if not os.path.islink(f'{outdir}/temp/{prefix}.scopen.k_{kl.knee}.weights.txt'):
        mv_symlink(f'{outdir}/temp/{prefix}.scopen.k_{kl.knee}.weights.txt', f'{outdir}/{prefix}.scopen.k_{kl.knee}.weights.txt')
    if not os.path.islink(f'{outdir}/temp/{prefix}.scopen.k_{kl.knee}.scores.txt'):
        mv_symlink(f'{outdir}/temp/{prefix}.scopen.k_{kl.knee}.scores.txt', f'{outdir}/{prefix}.scopen.k_{kl.knee}.scores.txt')
    log.log(f'Automatically selecting {kl.knee} components for {dataset} {prefix} based on knee point of reconstruction error', calling_file = 'run_scopen')
# endregion

@log.profile
def main(args):
    cistopic_outdir = os.path.dirname(proj.to_pathname('atac_programmes_cistopic', args.dataset, args.prefix))
    scopen_outdir = os.path.dirname(proj.to_pathname('atac_programmes_scopen', args.dataset, args.prefix))
    if args.cistopic:
        run_cistopic(args.dataset, args.prefix, cistopic_outdir, n_components = args.cistopic_components, worker = args.worker, niter = args.niter,
            alpha = args.cistopic_alpha, eta = args.cistopic_eta, mallet = args.mallet,
            cell_type = args.cell_type, embedding = args.embedding, time_key = args.time_key, seed = 19260817, downsample = args.downsample,
            force = args.force, savedir = args.magma_out if args.magma else None)
    if args.scopen:
        run_scopen(args.dataset, args.prefix, scopen_outdir, n_components = args.scopen_components,
            cell_type = args.cell_type, embedding = args.embedding, time_key = args.time_key, seed = 19260817,
            force = args.force, savedir = args.magma_out if args.magma else None)

def add_cmd_args(parser):
    parser.add_argument('--cistopic', action = 'store_true', help = 'Run cistopic to identify gene programmes')
    parser.add_argument('--cistopic_components', type = int, nargs = '+', default = [25, 500, 25],
        help = 'Number of components to identify for cistopic (start, stop, step), default: 25 to 500 in steps of 25')
    parser.add_argument('--cistopic_alpha', type = float, default = 50, help = 'Alpha parameter for cistopic (default: 50)')
    parser.add_argument('--cistopic_eta', type = float, default = 0.1, help = 'Eta parameter for cistopic (default: 0.1)')
    parser.add_argument('--mallet', default = '/rds/project/rds-Nl99R8pHODQ/toolbox/mallet/bin/mallet',
        help = 'Path to mallet binary for cistopic. default: /rds/project/rds-Nl99R8pHODQ/toolbox/mallet')

    parser.add_argument('--scopen', action = 'store_true', help = 'Run scOpen to identify gene programmes')
    parser.add_argument('--scopen_components', type = int, nargs = '+', default = [25, 500, 25],
        help = 'Number of components to identify for scOpen (start, stop, step), default: 25 to 500 in steps of 25')

    parser.add_argument('--downsample', type = int, default = -1, 
        help = 'Downsample cells to this number for cistopic/scOpen (default: -1 = auto, specify 0 for no downsampling)')
    parser.add_argument('--niter', type = int, default = 100, help = 'Number of iterations for cistopic/scOpen (default: 100)')
    parser.add_argument('--cell_type', type = str, nargs = '*', default = ['cell_type'],
        help = '''Categorical factors in adata.obs that denote the cell type. 
        Only the first is used for Spectra decomposition and pseudotime regression plots.
        All factors are used for factor importance scoring
        (default: Type_updated)''')
    parser.add_argument('--embedding', type = str, nargs = '*', default = ['X_umap', 'X_tsne'],
        help = 'Embedding representations in adata.obsm to use for factor embedding plots (default: X_umap X_tsne)')
    parser.add_argument('--time_key', type = str, nargs = '*', default = ['pseudotime', 'age', 'time'],
        help = 'Column names in adata.obs that denote time-related variables to use for pseudotime regression plots (default: pseudotime age time)')
    parser.add_argument('--magma', action = 'store_true', help = 'Format output for MAGMA GSEA')
    parser.add_argument('--magma_out', default = '../gene_score', help = 'Output directory for MAGMA formatted gene weights (default: ../gene_score)')
    parser.add_argument('-f','--force', action = 'store_true', help = 'Force overwrite')
    return parser

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'Identify gene expression programmes from scRNA-seq data')
    parser.add_argument('dataset', type = str, help = 'Dataset name')
    parser.add_argument('prefix', type = str, help = 'Prefix for output files')
    parser.add_argument('--worker', type = int, default = 0, help = 'Worker ID for parallel processing (default: 0)')
    parser = add_cmd_args(parser)
    args = parser.parse_args()
    if len(args.cistopic_components) == 3: 
        args.cistopic_components = range(args.cistopic_components[0], args.cistopic_components[1]+1, args.cistopic_components[2])
    elif len(args.cistopic_components) == 1: args.cistopic_components = [args.cistopic_components[0]]
    else: log.error('Invalid format for --cistopic_components, please provide either a single integer or three integers for start, stop, step')
    log.splash(args)
    main(args)