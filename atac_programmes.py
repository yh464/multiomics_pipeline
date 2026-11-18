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
import os, time, pickle, glob, re
from fnmatch import fnmatch
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from _utils.adatatools import subset_h5ad, downsample_adata
from _utils.slurm import get_slurm_config
n_cpu, mem = get_slurm_config() # memory in MB

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

def _cistopic_select_k(k_selection_metrics):
        k_selection_metrics[['arun_2010', 'cao_juan_2009']] *= -1
        k_selection_metrics_scaled = k_selection_metrics.copy()
        for col in ['arun_2010', 'cao_juan_2009', 'mimno_2011', 'loglikelihood']:
            k_selection_metrics_scaled[col] = (k_selection_metrics[col] - k_selection_metrics[col].min()) / (k_selection_metrics[col].max() - k_selection_metrics[col].min())
        # compare using just mimno and loglikelihood
        k_selection_metrics_scaled['score'] = k_selection_metrics_scaled[['mimno_2011', 'loglikelihood']].mean(axis = 1)
        k_optim = k_selection_metrics_scaled.loc[k_selection_metrics_scaled['score'].idxmax(), 'k']
        return k_optim

def _generate_cistopic_bin_topics(dataset, prefix, outdir, n_components = range(25, 501, 25), worker = 0, niter = 100, 
    alpha = 50, eta = 0.1, mallet = '/rds/project/rds-Nl99R8pHODQ/toolbox/mallet', downsample = -1,
    cell_type = ['cell_type'], embedding = ['X_umap', 'X_tsne'], time_key = ['pseudotime', 'age', 'time'],
    seed = 19260817, force = False, savedir = None):

    # check progress
    if os.path.isfile(f'{outdir}/{prefix}.cistopic.k_selection_metrics.txt') or len(n_components) == 1:
        if len(n_components) == 1: k_optim = n_components[0]
        else:
            k_selection_metrics = pd.read_csv(f'{outdir}/{prefix}.cistopic.k_selection_metrics.txt', sep = '\t')
            k_optim = _cistopic_select_k(k_selection_metrics)
        if os.path.isfile(f'{outdir}/{prefix}.cistopic.k_{k_optim}.binarised_topics.pkl') and \
            os.path.isfile(f'{outdir}/{prefix}.cistopic.dar.{cell_type[0]}.pkl') and \
            not force:
            log.log(f'Binarised topics for {dataset} {prefix} already exist, skipping cistopic model fitting', calling_file = 'run_cistopic')
            log.log(f'Loading binarised topics from {outdir}/{prefix}.cistopic.k_{k_optim}.binarised_topics.pkl', calling_file = 'run_cistopic')
            log.log(f'Loading DARs from {outdir}/{prefix}.cistopic.dar.{cell_type[0]}.pkl', calling_file = 'run_cistopic')
            with open(f'{outdir}/{prefix}.cistopic.k_{k_optim}.binarised_topics.pkl', 'rb') as f: binarised_topics = pickle.load(f)
            with open(f'{outdir}/{prefix}.cistopic.dar.{cell_type[0]}.pkl', 'rb') as f: dar = pickle.load(f)
            return binarised_topics, dar, k_optim

    import scanpy as sc
    from scipy import sparse
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
        os.environ['MALLET_MEMORY'] = f'{mem*0.9/1024:.0f}G'
        
        models = run_cgs_models_mallet(cistopic_obj, n_topics = [n_topics], n_iter = niter, n_cpu = n_cpu,
            alpha = alpha, eta = eta, alpha_by_topic = True, eta_by_topic = False,
            random_state = seed, save_path = f'{outdir}/temp', mallet_path = mallet,
            reuse_corpus = True, tmp_path = f'{outdir}/temp'
            )
        with open(f'{outdir}/temp/{prefix}.cistopic.k_{n_topics}.{niter}iter.model.pkl', 'wb') as f: pickle.dump(models[0], f)

    if len(check_cistopic_complete(dataset, prefix, n_components = n_components)) > 0:
        log.log('Waiting for other workers to finish cistopic model fitting', calling_file = 'run_cistopic')
        return None, None, None

    # if all workers have finished, clean cache files that cannot be reused
    for x in glob.glob(f'{outdir}/temp/*state.mallet.gz') + glob.glob(f'{outdir}/temp/*inferencer.mallet') + \
        glob.glob(f'{outdir}/temp/*doctopics.txt') + glob.glob(f'{outdir}/temp/*topickeys.txt'):
        if os.path.isfile(x): os.remove(x)

    # compare models
    if len(n_components) > 1:
        log.log(f'Comparing cistopic models for {dataset} {prefix}', calling_file = 'run_cistopic')
        model_list = []
        for n_topics in n_components:
            with open(f'{outdir}/temp/{prefix}.cistopic.k_{n_topics}.{niter}iter.model.pkl', 'rb') as f: model_list.append(pickle.load(f))

        k_selection_metrics = pd.concat([m.metrics.assign(k = m.n_topic) for m in model_list]).set_index('k').sort_index().reset_index()
        k_selection_metrics.columns = ['k','arun_2010', 'cao_juan_2009', 'mimno_2011', 'loglikelihood']
        k_selection_metrics.to_csv(f'{outdir}/{prefix}.cistopic.k_selection_metrics.txt', index = False, sep = '\t')

        # rescale all metrics to [0, 1] for comparison
        k_optim = _cistopic_select_k(k_selection_metrics)
        log.log(f'Automatically selecting {k_optim} topics for {dataset} {prefix} based on mimno and loglikelihood', calling_file = 'run_cistopic')
        log.warn(f'Please check {outdir}/{prefix}.cistopic.k_selection_metrics.pdf for details', calling_file = 'run_cistopic')
        
        # region: plot scaled metrics, similar to gene_programmes.py lines 423-447
        k_selection_metrics[['arun_2010', 'cao_juan_2009']] *= -1 # needed for plotting as these metrics are inverted
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
            plot_ax.set_xticks(np.arange(min(n_components), max(n_components)+1, max(4*(n_components[1] - n_components[0]), 5)))
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
    else: 
        k_optim = n_components[0]

    # export best model loadings and binarised sets
    with open(f'{outdir}/temp/{prefix}.cistopic.k_{k_optim}.{niter}iter.model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    if not os.path.isfile(f'{outdir}/{prefix}.cistopic.k_{k_optim}.topic_loadings.txt.gz') or force:
        best_model.topic_region.T.to_csv(f'{outdir}/{prefix}.cistopic.k_{k_optim}.topic_loadings.txt.gz', sep = '\t')
    if not os.path.isfile(f'{outdir}/{prefix}.cistopic.k_{k_optim}.topic_scores.txt') or force:
        cell_topic = best_model.cell_topic.T # rows are now cells, columns are topics
        # normalise such that each cell sums to one
        cell_topic = cell_topic.div(cell_topic.sum(axis = 1), axis = 0)
        cell_topic.to_parquet(f'{outdir}/{prefix}.cistopic.k_{k_optim}.cell_topic.parquet', compression = 'gzip')
    cistopic_obj.add_LDA_model(best_model)

    # identify DARs for each cell type annotation
    if not all([os.path.isfile(f'{outdir}/{prefix}.cistopic.dar.{ct}.pkl') for ct in cell_type]) or force:
        from pycisTopic.diff_features import impute_accessibility, normalize_scores, find_highly_variable_features, find_diff_features
        log.log(f'Identifying differentially accessible regions for {dataset} {prefix}', calling_file = 'run_cistopic')
        imputed_acc_obj = impute_accessibility(cistopic_obj, selected_cells = None, selected_regions = None, scale_factor = 1e6)
        norm_imputed_obj = normalize_scores(imputed_acc_obj, scale_factor = 1e6)
        hvp = find_highly_variable_features(norm_imputed_obj, min_disp = 0.05, min_mean = 0.0125, max_mean = 3, max_disp = np.inf, 
            n_bins = 20, n_top_features = None, plot = False)
        log.log(f'Found {len(hvp)} highly variable peaks for {dataset} {prefix}', calling_file = 'run_cistopic')
        for ct in tqdm(cell_type, desc = f'Identifying DARs for {dataset} {prefix}'):
            if os.path.isfile(f'{outdir}/{prefix}.cistopic.dar.{ct}.pkl') and not force: continue
            log.log(f'Identifying DARs for {ct} in {dataset} {prefix}', calling_file = 'run_cistopic')
            os.environ['RAY_memory_usage_threshold'] = '0.99'
            markers_dict = find_diff_features(cistopic_obj, imputed_acc_obj, var_features = hvp, 
                variable = ct, n_cpu = max(int(n_cpu/2), 1), split_pattern = '-',
                _temp_dir = '/rds/user/yh464/hpc-work/temp/ray') # must use a short path name as ray has a limit of 107 bytes for the tmpdir
            with open(f'{outdir}/{prefix}.cistopic.dar.{ct}.pkl', 'wb') as f: pickle.dump(markers_dict, f)
    dar = pickle.load(open(f'{outdir}/{prefix}.cistopic.dar.{cell_type[0]}.pkl', 'rb'))

    # binarise topics and annotate topics with cell types
    from pycisTopic.topic_binarization import binarize_topics
    from pycisTopic.topic_qc import topic_annotation
    binarised_topics = binarize_topics(cistopic_obj, method = 'otsu') 
    # returns a dict of f'Topic {i}': pd.DataFrame(topic_score, index = region_names, columns = [f'Topic {i}'])
    # convert to MAGMA format: Topic_{i} region_1 region_2 ... region_n
    
    with open(f'{outdir}/{prefix}.cistopic.k_{k_optim}.binarised_topics.pkl', 'wb') as f: pickle.dump(binarised_topics, f)

    with open(f'{outdir}/{prefix}.cistopic.k_{k_optim}.binarised_topics.txt', 'w') as f:
        for topic, peaks in binarised_topics.items():
            topic = topic.replace('Topic ', 'Topic_')
            f.write(f'{topic} ' + ' '.join(peaks.index.tolist()) + '\n')

    if not os.path.isfile(f'{outdir}/{prefix}.cistopic.k_{k_optim}.topic_annotation.txt') or force:
        log.log(f'Annotating topics for {dataset} {prefix} with {k_optim} topics', calling_file = 'run_cistopic')
        topic_annot = pd.concat([
            topic_annotation(cistopic_obj, annot_var = ct, binarized_cell_topic = binarised_topics, general_topic_thr = 0.2
                ).reset_index(names = 'topic').rename(columns = {ct: 'cell_type'}).assign(annotation = ct).iloc[:, [0,5,1,2,3,4]]
            for ct in cell_type
        ])
        topic_annot.to_csv(f'{outdir}/{prefix}.cistopic.k_{k_optim}.topic_annotation.txt', sep = '\t', index = False)

    return binarised_topics, dar, k_optim

def _run_cistarget_ctx(dataset, prefix, outdir, cell_type, binarised_topics_pr, dar_pr, ctx_db_path, motif_annot_path,
    species = 'homo_sapiens', auc_threshold = 0.005, nes_threshold = 3.0, rank_threshold = 0.05, annot_version = 'v10nr_clust', 
    annotation_to_use = None, motif_similarity_fdr = 0.001, orthologous_identity_threshold = 0.0, motifs_to_use = None, force = False):
    # default arguments are copied from pycistarget package default

    import dill
    topics_outfile = f'{outdir}/{prefix}.cistopic.k_{len(binarised_topics_pr)}.cistarget_ctx.pkl'
    dar_outfile = f'{outdir}/{prefix}.cistopic.dar.{cell_type}.cistarget_ctx.pkl'
    if os.path.isfile(topics_outfile) and os.path.isfile(dar_outfile) and not force:
        with open(topics_outfile, 'rb') as f: topics_results = dill.load(f)
        with open(dar_outfile, 'rb') as f: dar_results = dill.load(f)
        return topics_results, dar_results

    from pycistarget.motif_enrichment_cistarget import cisTargetDatabase, cisTarget
    log.log(f'Creating cisTarget database for {dataset} {prefix}', calling_file = 'run_cistopic')
    ctx_db = cisTargetDatabase(
        fname = f'{ctx_db_path}/hg38_screen_v10_clust.regions_vs_motifs.rankings.feather',
        region_sets = binarised_topics_pr | dar_pr, # merge two dicts when creating the database, but run and save results separately   
    )

    log.log(f'Running pycisTarget for {dataset} {prefix} using {len(binarised_topics_pr)} topics', calling_file = 'run_cistopic')
    if not os.path.isfile(topics_outfile) or force:
        topics_results = {
            cisTarget(
                region_set = peaks, name = f'{prefix}_k{len(binarised_topics_pr)}_{topic}', species = species,
                auc_threshold = auc_threshold, nes_threshold = nes_threshold, rank_threshold = rank_threshold, path_to_motif_annotations = motif_annot_path,
                annotation_version = annot_version, annotation_to_use = annotation_to_use, motif_similarity_fdr = motif_similarity_fdr,
                orthologous_identity_threshold = orthologous_identity_threshold, motifs_to_use = motifs_to_use
            ).run_ctx(ctx_db)
            for topic, peaks in binarised_topics_pr.items()
        }
        with open(topics_outfile, 'wb') as f: dill.dump(topics_results, f)
    else: 
        with open(topics_outfile, 'rb') as f: topics_results = dill.load(f)

    log.log(f'Running pycisTarget for {dataset} {prefix} using {len(dar_pr)} DARs across {cell_type}', calling_file = 'run_cistopic')
    if not os.path.isfile(dar_outfile) or force:
        dar_results = {
            cisTarget(
                region_set = peaks, name = f'{prefix}_dar_{cell_type}_{ct}', species = species,
                auc_threshold = auc_threshold, nes_threshold = nes_threshold, rank_threshold = rank_threshold, path_to_motif_annotations = motif_annot_path,
                annotation_version = annot_version, annotation_to_use = annotation_to_use, motif_similarity_fdr = motif_similarity_fdr,
                orthologous_identity_threshold = orthologous_identity_threshold, motifs_to_use = motifs_to_use
            ).run_ctx(ctx_db)
            for ct, peaks in dar_pr.items()
        }
        with open(dar_outfile, 'wb') as f: dill.dump(dar_results, f)
    else:
        with open(dar_outfile, 'rb') as f: dar_results = dill.load(f)

    return topics_results, dar_results

def _run_cistarget_dem(dataset, prefix, outdir, cell_type, binarised_topics_pr, dar_pr, background_pr, dem_db_path, motif_annot_path,
    species = 'homo_sapiens', adjpval_thr = 0.05, log2fc_thr = 1.0, mean_fg_thr = 0.0, motif_hit_thr = None, annot_version = 'v10nr_clust',
    annotation_to_use = None, motif_similarity_fdr = 0.001, orthologous_identity_threshold = 0.5, motifs_to_use = None, force = False):
    # default arguments are copied from pycistarget package default

    import dill
    import pyranges as pr
    topics_outfile = f'{outdir}/{prefix}.cistopic.k_{len(binarised_topics_pr)}.cistarget_dem.pkl'
    dar_outfile = f'{outdir}/{prefix}.cistopic.dar.{cell_type}.cistarget_dem.pkl'
    if os.path.isfile(topics_outfile) and os.path.isfile(dar_outfile) and not force:
        with open(topics_outfile, 'rb') as f: topics_results = dill.load(f)
        with open(dar_outfile, 'rb') as f: dar_results = dill.load(f)
        return topics_results, dar_results

    from pycistarget.motif_enrichment_dem import DEMDatabase, DEM
    log.log(f'Creating DEM database from {dem_db_path}')
    dem_db = DEMDatabase(dem_db_path)
    _, dem_scores = dem_db.get_scores( # merge all regions to read scores from memory
        pr.concat(list(binarised_topics_pr.values()) + list(dar_pr.values()) + [background_pr]))
    
    log.log(f'Running pycisTarget DEM for {dataset} {prefix} using {len(binarised_topics_pr)} topics', calling_file = 'run_cistopic')
    if not os.path.isfile(topics_outfile) or force:
        topics_results = {
            DEM(
                foreground_regions = peaks, background_regions = background_pr, name = f'{prefix}_k{len(binarised_topics_pr)}_{topic}', 
                species = species, adjpval_thr = adjpval_thr, log2fc_thr = log2fc_thr, mean_fg_thr = mean_fg_thr, motif_hit_thr = motif_hit_thr,
                path_to_motif_annotations = motif_annot_path, annotation_version = annot_version, annotation_to_use = annotation_to_use,
                motif_similarity_fdr = motif_similarity_fdr, orthologous_identity_threshold = orthologous_identity_threshold, motifs_to_use = motifs_to_use
            ).run(dem_db, dem_scores)
            for topic, peaks in binarised_topics_pr.items()
        }
        with open(topics_outfile, 'wb') as f: dill.dump(topics_results, f)
    else:
        with open(topics_outfile, 'rb') as f: topics_results = dill.load(f)

    log.log(f'Running pycisTarget DEM for {dataset} {prefix} using {len(dar_pr)} DARs across {cell_type}', calling_file = 'run_cistopic')
    if not os.path.isfile(dar_outfile) or force:
        dar_results = {
            DEM(
                foreground_regions = peaks, background_regions = background_pr, name = f'{prefix}_dar_{cell_type}_{ct}', 
                species = species, adjpval_thr = adjpval_thr, log2fc_thr = log2fc_thr, mean_fg_thr = mean_fg_thr, motif_hit_thr = motif_hit_thr,
                path_to_motif_annotations = motif_annot_path, annotation_version = annot_version, annotation_to_use = annotation_to_use,
                motif_similarity_fdr = motif_similarity_fdr, orthologous_identity_threshold = orthologous_identity_threshold, motifs_to_use = motifs_to_use
            ).run(dem_db, dem_scores)
            for ct, peaks in dar_pr.items()
        }
        with open(dar_outfile, 'wb') as f: dill.dump(dar_results, f)
    else:
        with open(dar_outfile, 'rb') as f: dar_results = dill.load(f)

    return topics_results, dar_results

def run_cistopic(dataset, prefix, outdir, n_components = range(25, 501, 25), 
    worker = 0, niter = 100, top = 3000, downsample = -1, alpha = 50, eta = 0.1, 
    scenic_db = '/rds/project/rds-Nl99R8pHODQ/toolbox/scenicplus/databases', 
    mallet = '/rds/project/rds-Nl99R8pHODQ/toolbox/mallet',
    cell_type = ['cell_type'], embedding = ['X_umap', 'X_tsne'], time_key = ['pseudotime', 'age', 'time'],
    seed = 19260817, force = False, savedir = None):

    binarised_topics, dar, k_optim = _generate_cistopic_bin_topics(dataset, prefix, outdir, n_components = n_components, worker = worker, niter = niter,
        alpha = alpha, eta = eta, mallet = mallet, downsample = downsample, cell_type = cell_type, embedding = embedding, time_key = time_key, 
        seed = seed, force = force, savedir = savedir)
    if binarised_topics is None: return

    # binarise topics using top peaks if requested
    log.log(f'Binarising topics using top {top} peaks for {dataset} {prefix}', calling_file = 'run_cistopic')
    if top > 0 and (not os.path.isfile(f'{outdir}/{prefix}.cistopic.k_{k_optim}.binarised_topics.top{top}.pkl') or force):
        peak_scores = pd.read_table(f'{outdir}/{prefix}.cistopic.k_{k_optim}.topic_loadings.txt.gz', index_col = 0)
        with open(f'{outdir}/{prefix}.cistopic.k_{k_optim}.binarised_topics.top{top}.pkl', 'wb') as f:
            pickle.dump({topic: peaks.sort_values(ascending = False).head(top) for topic, peaks in peak_scores.items()}, f)

    # convert the binarized topics to a dict of pyRanges objects
    import pyranges as pr
    binarised_topics_pr = {topic: pr.PyRanges(
        chromosomes = peaks.index.str.split(':|-').str[0],
        starts = peaks.index.str.split(':|-').str[1].astype(int),
        ends = peaks.index.str.split(':|-').str[2].astype(int))
        for topic, peaks in binarised_topics.items()}
    dar_pr = {re.sub('[^A-Za-z0-9]+', '_', ct): pr.PyRanges(
        chromosomes = peaks.index.str.split(':|-').str[0],
        starts = peaks.index.str.split(':|-').str[1].astype(int),
        ends = peaks.index.str.split(':|-').str[2].astype(int)
    ) for ct, peaks in dar.items()}

    # pyGREAT pipeline
    # convert the binarized topics to a dict of pyRanges objects
    # currently not implemented because the pyGREAT server http://great.stanford.edu/public/cgi-bin/greatWeb.php is down
    # from pycisTopic.pyGREAT import pyGREAT
    # topics_pygreat = pyGREAT(binarised_topics_pr, species = 'hg38')
    # # returns a dict {Topic i: {ontology: pd.DataFrame()}}
    # topics_pygreat_df = pd.concat([
    #     pd.concat([df.assign(ontology = ontology) for ontology, df in topic_dict.items()]
    #     ).assign(topic = topic) for topic, topic_dict in topics_pygreat.items()])
    # topics_pygreat_df.to_csv(f'{outdir}/{prefix}.cistopic.k_{k_optim}.pyGREAT.txt', sep = '\t', index = False)

    # pycisTarget pipeline
    topics_ctx, dar_ctx = _run_cistarget_ctx(dataset, prefix, outdir, cell_type[0], binarised_topics_pr, dar_pr,
        ctx_db_path = f'{scenic_db}/hg38_screen_v10_clust.regions_vs_motifs.rankings.feather',
        motif_annot_path = f'{scenic_db}/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl',
        force = force) # use default arguments for pycisTarget, which are copied from pycistarget package default

    # background regions are all peaks in the dataset
    import h5py
    with h5py.File(proj.to_pathname('raw', dataset, prefix), 'r') as f:
        var_names = pd.Series(f['var'][f['var'].attrs['_index']][:].astype(str))
        background_pr = pr.PyRanges(
            chromosomes = var_names.str.split(':|-').str[0],
            starts = var_names.str.split(':|-').str[1].astype(int),
            ends = var_names.str.split(':|-').str[2].astype(int)
        )
    topics_dem, dar_dem = _run_cistarget_dem(dataset, prefix, outdir, cell_type[0], binarised_topics_pr, dar_pr, background_pr,
        dem_db_path = f'{scenic_db}/hg38_screen_v10_clust.regions_vs_motifs.scores.feather',
        motif_annot_path = f'{scenic_db}/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl',
        force = force) # use default arguments for pycisTarget, which are copied from pycistarget package default

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
        sc.pp.filter_cells(adata, min_counts = 1)
        sc.pp.filter_genes(adata, min_cells = 1)

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
        W = pd.DataFrame(nmf_output[0], index = adata_tfidf.var_names, columns = [f'F{i}' for i in range(1, k + 1)])
        H = pd.DataFrame(nmf_output[1], index = [f'F{i}' for i in range(1, k + 1)], columns = adata_tfidf.obs_names)
        error = nmf_output[2]
        W.to_csv(f'{outdir}/temp/{prefix}.scopen.k_{k}.weights.txt', sep = '\t')
        H.to_csv(f'{outdir}/temp/{prefix}.scopen.k_{k}.scores.txt', sep = '\t')
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
            alpha = args.cistopic_alpha, eta = args.cistopic_eta, mallet = args.mallet, scenic_db = args.scenic_db,
            cell_type = args.cell_type, embedding = args.embedding, time_key = args.time_key, seed = 19260817, downsample = args.downsample,
            force = args.force, savedir = args.magma_out if args.magma else None, top = args.top)
    if args.scopen:
        run_scopen(args.dataset, args.prefix, scopen_outdir, n_components = args.scopen_components, worker = args.worker,
            cell_type = args.cell_type, embedding = args.embedding, time_key = args.time_key, seed = 19260817,
            force = args.force, savedir = args.magma_out if args.magma else None, top = args.top)

def add_cmd_args(parser):
    parser.add_argument('--cistopic', action = 'store_true', help = 'Run cistopic to identify gene programmes')
    parser.add_argument('--cistopic_components', type = int, nargs = '+', default = [25, 500, 25],
        help = 'Number of components to identify for cistopic (start, stop, step), default: 25 to 500 in steps of 25')
    parser.add_argument('--cistopic_alpha', type = float, default = 50, help = 'Alpha parameter for cistopic (default: 50)')
    parser.add_argument('--cistopic_eta', type = float, default = 0.1, help = 'Eta parameter for cistopic (default: 0.1)')
    parser.add_argument('--mallet', default = '/rds/project/rds-Nl99R8pHODQ/toolbox/mallet/bin/mallet',
        help = 'Path to mallet binary for cistopic. (default: /rds/project/rds-Nl99R8pHODQ/toolbox/mallet)')
    parser.add_argument('--scenic_db', help = 'SCENIC+ database directory for pycisTarget; KEEP DEFAULT FILE NAMES',
        default = '/rds/project/rds-Nl99R8pHODQ/toolbox/scenicplus/databases')

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
        (default: cell_type)''')
    parser.add_argument('--embedding', type = str, nargs = '*', default = ['X_umap', 'X_tsne'],
        help = 'Embedding representations in adata.obsm to use for factor embedding plots (default: X_umap, X_tsne)')
    parser.add_argument('--top', type = int, default = 3000, 
        help = 'Number of top peaks to use when binarising factors (default: 3000, set --top 0 to disable)')
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
    if len(args.scopen_components) == 3:
        args.scopen_components = range(args.scopen_components[0], args.scopen_components[1]+1, args.scopen_components[2])
    elif len(args.scopen_components) == 1: args.scopen_components = [args.scopen_components[0]]
    else: log.error('Invalid format for --scopen_components, please provide either a single integer or three integers for start, stop, step')
    log.splash(args)
    main(args)