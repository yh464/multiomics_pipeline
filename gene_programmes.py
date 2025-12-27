#!/usr/bin/env python3
'''
Author: Yuankai He
Correspondence: yh464@cam.ac.uk
Version 1: 2025-12-21

Pipelines to identify gene expression programmes from scRNA-seq data
'''

import anndata
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, time
from fnmatch import fnmatch
from multiprocessing import cpu_count
from _utils.logger import logger
log = logger()
from _utils.path import project
proj = project()


def run_cnmf(dataset, prefix, outdir, n_components = range(10, 71, 10), seed = 19260817, force = False, worker_id = 0, n_iter = 100):
    '''
    run consensus NMF on input h5ad file (RAW COUNTS)
    outdir: output directory
    n_components: number of components to identify
    random_state: random seed
    '''

    h5ad_raw = proj.to_pathname('raw', dataset, prefix)
    outdir = os.path.realpath(outdir).replace('$dataset', dataset).replace('$prefix', prefix)

    import cnmf
    log.log(f'Conducting cNMF on {h5ad_raw}', calling_file = 'run_cnmf')
    log.log(f'Output directory: {outdir}/{prefix}', calling_file = 'run_cnmf')
    cnmf_obj = cnmf.cNMF(output_dir = outdir, name = prefix)
    while not os.path.isfile(cnmf_obj.paths['normalized_counts']) or not os.path.isfile(cnmf_obj.paths['tpm']):
        if worker_id == 0: # prevent other workers from simultaneously writing files
            cnmf_obj.prepare(counts_fn = h5ad_raw, components = n_components, n_iter = n_iter, seed = seed)
            cnmf_obj.update_nmf_iter_params()
        else: time.sleep(10)
    if not force: skip_completed = True
    else: skip_completed = False
    cnmf_obj.factorize(worker_i = worker_id, total_workers = 100, skip_completed_runs = skip_completed)
    n_spectra_complete = 0
    for f in os.listdir(f'{outdir}/{prefix}/cnmf_tmp'):
        for k in n_components:
            if fnmatch(f, f'{prefix}.spectra.k_{k}.iter_*.df.npz'): n_spectra_complete += 1
    log.log(f'Completed {n_spectra_complete} / {len(n_components)*n_iter} spectra matrix factorizations', calling_file = 'run_cnmf')
    if n_spectra_complete < len(n_components)*n_iter:
        log.warn('Waiting for other workers to complete iterations', calling_file = 'run_cnmf')
        return
    cnmf_obj.combine()
    cnmf_obj.k_selection_plot()
    k_selection_stats = np.load(cnmf_obj.paths['k_selection_stats'])
    k_optim = k_selection_stats.k[k_selection_stats.silhouette.argmax()] # NEED TO DOUBLE CHECK ON THE PLOTS, ONLY A GUIDE
    log.log(f'Optimal number of components identified: {k_optim}', calling_file = 'run_cnmf')
    cnmf_obj.consensus(k = k_optim, density = 0.01)

def run_spectra(dataset, prefix, outdir):
    '''
    run Spectra on input h5ad file (RAW COUNTS)
    outdir: output directory
    n_components will be estimated from data
    '''
    
    h5ad_raw = proj.to_pathname('raw', dataset, prefix)
    adata = sc.read_h5ad(h5ad_raw)
    adata.X = adata.X.log1p() # convert to log counts
    outdir = os.path.realpath(outdir).replace('$dataset', dataset).replace('$prefix', prefix)

    import Spectra
    annotations = Spectra.default_gene_sets.load()

    pass

def run_scired(dataset, prefix, outdir, n_components = 50, n_genes = 2000, 
    covar_cols = [], factors_to_explain = [],           
    seed = 19260817, force = False):
    '''
    run consensus NMF on input h5ad file
    h5ad: input h5ad file path or AnnData object, RAW COUNTS
    outdir: output directory
    n_components: number of components to identify
    random_state: random seed
    '''
    h5ad_raw = proj.to_pathname('raw', dataset, prefix)
    adata = sc.read_h5ad(h5ad_raw)
    outdir = os.path.realpath(outdir).replace('$dataset', dataset).replace('$prefix', prefix)
    
    # output files for progress checking
    os.makedirs(outdir, exist_ok = True)
    out_loading = f'{outdir}/{prefix}_scired_loadings.txt'
    out_scores = f'{outdir}/{prefix}_scired_scores.txt'
    out_fcat = f'{outdir}/{prefix}_scired_fcat.txt'
    out_fcat_fig = f'{outdir}/{prefix}_scired_fcat.pdf'
    out_interpretability = f'{outdir}/{prefix}_scired_interpretability.txt'
    out_interpretability_fig = f'{outdir}/{prefix}_scired_interpretability.pdf'

    import sciRED
    import sciRED.utils
    import sciRED.ensembleFCA
    import statsmodels.api as sm
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from _plots.corr_heatmap import corr_heatmap
    from _plots.colourcode_scatterplot import scatterplot_adata
    log.log(f'Conducting scIRED on input AnnData object', calling_file = 'run_scired')
    log.log(f'Output directory: {outdir}', calling_file = 'run_scired')
    log.log(f'Number of components to identify: {n_components}', calling_file = 'run_scired')
    log.log(f'Number of HV genes to use: {n_genes}', calling_file = 'run_scired')
    log.log(f'Covariates to adjust for: {", ".join(covar_cols)}', calling_file = 'run_scired')
    np.random.seed(seed)

    adata, gene_idx = sciRED.utils.preprocess.get_sub_data(adata, num_genes = n_genes)
    y, genes, num_cells, num_genes = sciRED.utils.preprocess.get_data_array(adata)
    adata.obs['n_umi'] = adata.X.sum(axis = 1)

    if os.path.isfile(out_loading) and os.path.isfile(out_scores) and not force:
        y_varimax = pd.read_table(out_scores, index_col = 0)
        loading_varimax = pd.read_table(out_loading, index_col = 0)
        log.log(f'Found existing scIRED output files, loading from {outdir}', calling_file = 'run_scired')
    else:
        # get covariates
        if 'n_umi' not in covar_cols: covar_cols.append('n_umi')
        design_mat = []
        for col in covar_cols:
            if col not in adata.obs.columns:
                raise ValueError(f'Covariate column {col} not found in adata.obs')
            if pd.api.types.is_categorical_dtype(adata.obs[col]) or adata.obs[col].dtype == object:
                dummies = pd.get_dummies(adata.obs[col], prefix = col, drop_first = True)
                design_mat.append(dummies)
            else:
                design_mat.append(adata.obs[[col]])
        design_mat = pd.concat(design_mat, axis = 1)
        design_mat = sm.add_constant(design_mat, has_constant = 'add')

        # factor identification for sciRED
        glm_fit_dict = sciRED.glm.poissonGLM(y, design_mat.values)
        resid_pearson = glm_fit_dict['resid_pearson']
        y = resid_pearson.T
        pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=n_components))])
        y_pca = pipeline.fit_transform(y)
        loading = pipeline.named_steps['pca'].components_.T
        rot_varimax = sciRED.rotations.varimax(loading)
        loading_varimax = rot_varimax['rotloading']
        y_varimax = sciRED.rotations.get_rotated_scores(y_pca, rot_varimax['rotmat'])
        
        # these variables are only used for output and should not affect downstream preprocessing
        loading_varimax_ = pd.DataFrame(loading_varimax, index = genes, columns = [f'F{i+1}' for i in range(loading_varimax.shape[1])])
        y_varimax_ = pd.DataFrame(y_varimax, index = adata.obs_names, columns = [f'F{i+1}' for i in range(y_varimax.shape[1])])
        loading_varimax_.to_csv(out_loading, sep = '\t', index = True, header = True)
        y_varimax_.to_csv(out_scores, sep = '\t', index = True, header = True)

        # plot UMAP scatterplot for all factors
        os.makedirs(f'{outdir}/plots', exist_ok = True)
        for factor in y_varimax_.columns:
            if 'X_umap' not in adata.obsm.keys(): continue
            fig = scatterplot_adata(adata, v = y_varimax_[factor], rep = 'umap')
            fig.savefig(f'{outdir}/plots/{prefix}_scired_{factor}_umap.pdf', bbox_inches = 'tight')
            plt.close(fig)

    # FCAT analysis for factor importance
    fcat_mat = []
    for col in factors_to_explain:
        if col not in adata.obs.columns:
            log.log(f'Factor to explain {col} not found in adata.obs, skipping.', calling_file = 'run_scired', warning = True)
            continue
        elif pd.api.types.is_categorical_dtype(adata.obs[col]) or adata.obs[col].dtype == object:
            fcat_col = sciRED.ensembleFCA.FCAT(adata.obs[col],  
                y_varimax, scale = 'standard', mean = 'arithmatic') # author spelling is incorrect
            fcat_col['explained_factor'] = fcat_col.index
            fcat_col = fcat_col.dropna().melt(id_vars = 'explained_factor', var_name = 'scired_factor', value_name = 'fcat_value')
            fcat_col.insert(0, 'explained_group', col)
            fcat_col.insert(2, 'xlabel', 'sciRED Factor')
            fcat_mat.append(fcat_col)
        else:
            log.log(f'Factor to explain {col} is not categorical, skipping.', calling_file = 'run_scired', warning = True)
    if len(fcat_mat) == 0: raise ValueError('No valid factors to explain provided.')
    fcat_mat = pd.concat(fcat_mat, axis = 0)
    fcat_thr = sciRED.ensembleFCA.get_otsu_threshold(fcat_mat['fcat_value'].dropna().values)
    fcat_mat['significance'] = fcat_mat['fcat_value'] >= fcat_thr
    fcat_mat.to_csv(out_fcat, sep = '\t', index = True, header = True)
    fig = corr_heatmap(fcat_mat, sort = False, sig_col = 'significance')
    fig.savefig(out_fcat_fig, bbox_inches = 'tight')
    plt.close(fig)

    # correlation with n_umi
    corr_numi = sciRED.utils.corr.get_factor_libsize_correlation(y_varimax, adata.obs['n_umi'].values)
    log.log(f'Max correlation between identified factors and library size (n_umi): {corr_numi.abs().max():.4f}', calling_file = 'run_scired')

    # interpretability scoring
    interpretability_metrics = pd.DataFrame(index = [f'F{i+1}' for i in range(y_varimax.shape[1])], columns = [])
    silhouette_score = sciRED.metrics.kmeans_bimodal_score(y_varimax, time_eff = True)
    bimodality_index = sciRED.metrics.bimodality_index(y_varimax)
    interpretability_metrics['bimodality_score'] = (np.array(silhouette_score) + np.array(bimodality_index)) / 2
    interpretability_metrics['effect_size'] = sciRED.metrics.factor_variance(y_varimax)
    interpretability_metrics['specificity_score'] = sciRED.metrics.simpson_diversity_index(
        fcat_mat.pivot(index = 'explained_factor', columns = 'scired_factor', values = 'fcat_value')
    )
    for col in fcat_col.explained_group.unique():
        interpretability_metrics[f'homogeneity_{col}'] = sciRED.metrics.average_scaled_var(
            y_varimax, covariate_vector = adata.obs[col].values, mean_type = 'arithmetic') # spelling is correct for this function
    interpretability_metrics.to_csv(out_interpretability, sep = '\t', index = True, header = True)

    interpretability_metrics = interpretability_metrics.reset_index().melt(id_vars = 'index', var_name = 'metric', value_name = 'value')
    interpretability_metrics = interpretability_metrics.rename(columns = {'index': 'scired_factor'})
    interpretability_metrics.insert(0, 'xlabel', 'sciRED Factor')
    interpretability_metrics.insert(2, 'ylabel', '')
    fig = corr_heatmap(interpretability_metrics, sort = False)
    fig.savefig(out_interpretability_fig, bbox_inches = 'tight')
    plt.close(fig)

def main(args):
    cnmf_outdir = os.path.dirname(os.path.dirname(proj.config['programmes_cnmf'])).replace(
        '$dataset', args.dataset).replace('$prefix', args.prefix) # cNMF automatically creates the $prefix subdirectory
    scired_outdir = os.path.dirname(proj.config['programmes_scired']).replace('$dataset', args.dataset).replace('$prefix', args.prefix)
    if args.cnmf:
        run_cnmf(args.dataset, args.prefix, cnmf_outdir, n_components = args.cnmf_components, force = args.force, worker_id = args.worker)
    if args.scired:
        run_scired(args.dataset, args.prefix, scired_outdir, n_components = args.scired_components,
            n_genes = args.scired_genes, covar_cols = args.scired_covars,
            factors_to_explain = args.cell_type, force = args.force)
        
def add_cmd_args(parser):
    parser.add_argument('--cnmf', action = 'store_true', help = 'Run consensus NMF to identify gene programmes')
    parser.add_argument('--cnmf_components', type = int, nargs = 3, default = (10, 71, 10),
        help = 'Number of components to identify for cNMF (start, stop, step), default: 10 71 10')
    parser.add_argument('--scired', action = 'store_true', help = 'Run scIRED to identify gene programmes')
    parser.add_argument('--scired_components', type = int, default = 50,
        help = 'Number of components to identify for scIRED (default: 50)')
    parser.add_argument('--scired_genes', type = int, default = 2000,
        help = 'Number of highly variable genes to use for scIRED (default: 2000)')
    parser.add_argument('--scired_covars', type = str, nargs = '+', default = ['sex'],
        help = 'Covariate columns in adata.obs to adjust for in scIRED (default: sex)')
    parser.add_argument('--cell_type', type = str, nargs = '+', default = ['Type_updated'],
        help = '''Categorical factors in adata.obs that denote the cell type. 
        First argument is used for both Spectra and sciRED interpretability analysis, 
        subsequent args only for sciRED (default: Type_updated)''')
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
    args.cnmf_components = range(args.cnmf_components[0], args.cnmf_components[1]+1, args.cnmf_components[2])
    log.splash(args)
    main(args)