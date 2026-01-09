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
from tqdm import tqdm
from _utils.logger import logger
log = logger()
from _utils.path import project
proj = project()
from _plots.corr_heatmap import corr_heatmap, corr_heatmap_wide_format
from _plots.colourcode_scatterplot import scatterplot_adata
from _plots.regplot import temporal_regplot
from _utils.enrichr import enrichr_continuous

def factor_enrichr(scores, top_negative = True, top = 200):
    out = []
    for col in scores.columns:
        log.log(f'Enrichr analysis for factor {col}', calling_file = 'factor_enrichr')
        enrichr_res = enrichr_continuous(scores, by = col, top = top, top_negative = top_negative, 
            use_background = False, silent = True)
        enrichr_res.insert(0, 'factor', col)
        out.append(enrichr_res)
    out = pd.concat(out, axis = 0)
    return out

def factor_importance(scores, adata, factors_to_explain, out_tabular, out_fig):
    # FCAT analysis for factor importance
    import sciRED.ensembleFCA
    fcat_mat = []
    for col in factors_to_explain:
        if col not in adata.obs.columns:
            log.log(f'Factor to explain {col} not found in adata.obs, skipping.', calling_file = 'factor_importance', warning = True)
            continue
        elif isinstance(adata.obs[col].dtype, pd.CategoricalDtype) or adata.obs[col].dtype == object:
            fcat_col = sciRED.ensembleFCA.FCAT(adata.obs[col],  
                scores, scale = 'standard', mean = 'arithmatic') # author spelling is incorrect
            fcat_col['explained_factor'] = fcat_col.index
            fcat_col = fcat_col.dropna().melt(id_vars = 'explained_factor', var_name = 'scired_factor', value_name = 'fcat_value')
            # the ensembleFCA function will automatically name the column as 'scired_factor' even if the input is not from sciRED
            fcat_col.insert(0, 'explained_group', col)
            fcat_col.insert(2, 'xlabel', 'factor')
            fcat_mat.append(fcat_col)
        else:
            log.log(f'Factor to explain {col} is not categorical, skipping.', calling_file = 'factor_importance', warning = True)
    if len(fcat_mat) == 0: raise ValueError('No valid factors to explain provided.')
    fcat_mat = pd.concat(fcat_mat, axis = 0)
    fcat_thr = sciRED.ensembleFCA.get_otsu_threshold(fcat_mat['fcat_value'].dropna().values)
    fcat_mat['significance'] = fcat_mat['fcat_value'] >= fcat_thr
    fcat_mat.rename(columns={'scired_factor': 'factor'}).to_csv(out_tabular, sep = '\t', index = True, header = True)
    fig = corr_heatmap(fcat_mat, sort = False, sig_col = 'significance')
    fig.savefig(out_fig, bbox_inches = 'tight')
    plt.close(fig)
    return fcat_mat

def factor_correlation(loadings, out_tabular, out_fig):
    from scipy.cluster.hierarchy import linkage, dendrogram
    loadings.columns = [f'F{i+1}' for i in range(loadings.shape[1])]
    linkage_matrix = linkage(loadings.T, method = 'average', metric = 'correlation')
    dendrogram_res = dendrogram(linkage_matrix, no_plot = True)
    loadings = loadings.iloc[:, dendrogram_res['leaves']]
    corr_mat = loadings.corr()
    log.log(f'Highest positive correlation between factors: {corr_mat.values.max():.4f}', calling_file = 'factor_correlation')
    log.log(f'Highest negative correlation between factors: {corr_mat.values.min():.4f}', calling_file = 'factor_correlation')
    corr_mat.to_csv(out_tabular, sep = '\t', index = True, header = True)
    fig = corr_heatmap_wide_format(corr_mat)
    fig.savefig(out_fig, bbox_inches = 'tight')
    plt.close(fig)
    return corr_mat

def factor_pseudotime_reg(scores, adata, out_fig, cell_type_key, pseudotime_key = 'pseudotime'):
    os.makedirs(os.path.dirname(out_fig), exist_ok = True)
    scores.columns = [f'F{i+1}' for i in range(scores.shape[1])]
    score_cols = scores.columns.tolist()
    scores = pd.concat([scores, adata.obs[[cell_type_key, pseudotime_key]]], axis = 1).dropna()
    for col in tqdm(score_cols, desc = 'Plotting factor pseudotime regression'):
        fig = temporal_regplot(scores, x = pseudotime_key, y = col, hue = cell_type_key)
        fig.savefig(out_fig.replace('$factor', col), bbox_inches = 'tight')
        plt.close(fig)

def run_cnmf(dataset, prefix, outdir, n_components = range(10, 71, 10), cell_type = [],
    seed = 19260817, force = False, worker_id = 0, n_iter = 100):
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

    # check progress and preprocess data
    if worker_id == 0: 
        log.log('Setting cNMF runtime parameters', calling_file = 'run_cnmf')
        replicate_params, run_params = cnmf_obj.get_nmf_iter_params(
            ks = n_components, n_iter = n_iter, random_state_seed = seed,
            beta_loss = 'frobenius', init = 'random', alpha_usage = 0.0, alpha_spectra = 0.0, max_iter = 1000
        )
        cnmf_obj.save_nmf_iter_params(replicate_params, run_params)
        cnmf_obj.update_nmf_iter_params()
        log.log('Saved cNMF runtime parameters', calling_file = 'run_cnmf')
    else: time.sleep(5)
    while not os.path.isfile(cnmf_obj.paths['normalized_counts']) or not os.path.isfile(cnmf_obj.paths['tpm']):
        if worker_id == 0: # prevent other workers from simultaneously writing files
            cnmf_obj.prepare(counts_fn = h5ad_raw, components = n_components, n_iter = n_iter, seed = seed)
        else: time.sleep(10)
    
    # run NMF iterations
    if not force: skip_completed = True
    else: skip_completed = False
    cnmf_obj.factorize(worker_i = worker_id, total_workers = 100, skip_completed_runs = skip_completed)
    n_spectra_complete = 0
    for f in os.listdir(f'{outdir}/{prefix}/cnmf_tmp'):
        for k in n_components:
            if fnmatch(f, f'{prefix}.spectra.k_{k}.iter_*.df.npz'): n_spectra_complete += 1
    log.log(f'Completed {n_spectra_complete} / {len(n_components)*n_iter} NMF iterations', calling_file = 'run_cnmf')
    if n_spectra_complete < len(n_components)*n_iter:
        log.warn('Waiting for other workers to complete iterations', calling_file = 'run_cnmf')
        return
    
    # combine iterations
    for k in n_components:
        if os.path.isfile(cnmf_obj.paths['merged_spectra'] % k) and not force: continue
        cnmf_obj.combine_nmf(k)
    if len(n_components) > 1:
        cnmf_obj.k_selection_plot()
        os.rename(cnmf_obj.paths['k_selection_plot'], cnmf_obj.paths['k_selection_plot'].replace(
            '.png', f'_{min(n_components)}_{max(n_components)}_{int(n_components[1]-n_components[0])}.png'))
        with np.load(cnmf_obj.paths['k_selection_stats'], allow_pickle = True) as file:
            k_selection_stats = pd.DataFrame(**file)
        k_optim = k_selection_stats.k.astype(int)[k_selection_stats.silhouette.argmax()] # NEED TO DOUBLE CHECK ON THE PLOTS, ONLY A GUIDE
        log.log(f'Optimal number of components identified: {k_optim}', calling_file = 'run_cnmf')
    else: 
        k_optim = n_components[0]
        log.log(f'Proceeding with k = {k_optim} for downstream analysis', calling_file = 'run_cnmf')

    # consensus factor decomposition
    if not os.path.isfile(cnmf_obj.paths['consensus_spectra__txt'].replace(r'%d', str(k_optim)).replace(r'%s', '0_01')) or args.force:
        cnmf_obj.consensus(k = k_optim, density_threshold = 0.01)
    plot_dir = f'{outdir}/{prefix}/k_{k_optim}_plots'
    os.makedirs(plot_dir, exist_ok = True)
    usages = pd.read_table(cnmf_obj.paths['consensus_usages__txt'].replace(r'%d', str(k_optim)).replace(r'%s', '0_01'), index_col = 0)
    adata = sc.read_h5ad(h5ad_raw, 'r')
    for component in tqdm(usages.columns.tolist(), desc = 'Plotting cNMF cell-level scores in UMAP space'):
        if 'X_umap' not in adata.obsm.keys(): continue
        fig = scatterplot_adata(adata, v = usages[component], rep = 'umap')
        fig.savefig(f'{plot_dir}/{prefix}_cnmf_k{k_optim}_f{component}.png', bbox_inches = 'tight', dpi = 400)
        plt.close(fig)

    # pseudotime regression plots
    if len(cell_type) > 0 and 'pseudotime' in adata.obs.columns:
        factor_pseudotime_reg(usages, adata, f'{plot_dir}/{prefix}_cnmf_k{k_optim}_$factor_pseudotime.png', cell_type[0])
    
    # factor importance scoring
    out_fcat = f'{outdir}/{prefix}/{prefix}_cnmf_k{k_optim}_fcat.txt'
    out_fcat_fig = f'{outdir}/{prefix}/{prefix}_cnmf_k{k_optim}_fcat.pdf'
    factor_importance(usages.values, adata, cell_type, out_fcat, out_fcat_fig)
    log.log(f'cNMF factor importance analysis saved to {out_fcat} and {out_fcat_fig}', calling_file = 'run_cnmf')

    # correlation and enrichment analysis
    out_corr = f'{outdir}/{prefix}/{prefix}_cnmf_k{k_optim}_correlation.txt'
    out_corr_fig = f'{outdir}/{prefix}/{prefix}_cnmf_k{k_optim}_correlation.pdf'
    out_enrichr = f'{outdir}/{prefix}/{prefix}_cnmf_k{k_optim}_enrichr.txt'
    loadings = pd.read_table(cnmf_obj.paths['consensus_spectra__txt'].replace(r'%d', str(k_optim)).replace(r'%s', '0_01'), index_col = 0).T
    factor_correlation(loadings, out_corr, out_corr_fig)
    enrichr_res = factor_enrichr(loadings, top_negative = False)
    enrichr_res.to_csv(out_enrichr, sep = '\t', index = False, header = True)
    log.log(f'cNMF factor enrichment analysis saved to {out_enrichr}', calling_file = 'run_cnmf')
    
    proj.complete_step('programmes_cnmf', dataset, prefix)
    proj.complete_step('programmes_cnmf_scores', dataset, prefix)

def check_cnmf_completed(dataset, prefix, outdir, n_components = range(10, 71, 10), n_iter = 100):
    '''check if cNMF has been completed for given dataset / prefix'''
    outdir = os.path.realpath(outdir).replace('$dataset', dataset).replace('$prefix', prefix)
    outdir = os.path.dirname(outdir)  # remove $prefix to get to the parent directory
    n_spectra_complete = 0
    if not os.path.isdir(f'{outdir}/{prefix}/cnmf_tmp'): os.makedirs(f'{outdir}/{prefix}/cnmf_tmp')
    for f in os.listdir(f'{outdir}/{prefix}/cnmf_tmp'):
        for k in n_components:
            if fnmatch(f, f'{prefix}.spectra.k_{k}.iter_*.df.npz'): n_spectra_complete += 1
    if n_spectra_complete < len(n_components)*n_iter:
        return False
    return True

def run_spectra(dataset, prefix, outdir, cell_type):
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
    model = Spectra.est_spectra(
        adata = adata,
        gene_set_dictionary = annotations,
        use_highly_variable = True,
        cell_type_key = cell_type,
        lam = 0.1,
        delta = 0.001,
        kappa = None,
        rho = 0.001,
        use_cell_types = True,
        n_top_vals = 50,
        label_factors = True,
        overlap_threshold = 0.2, 
        clean_gs = True,
        min_gs_num = 3,
        num_epochs = 10000
    )

    adata.uns['SPECTRA_factors'].to_csv(f'{outdir}/{prefix}_spectra_loadings.txt', sep = '\t', index = True, header = True)
    adata.uns['SPECTRA_markers'].to_csv(f'{outdir}/{prefix}_spectra_markers.txt', sep = '\t', index = True, header = True)
    cell_scores = adata.obsm['SPECTRA_cell_scores']
    cell_scores = pd.DataFrame(cell_scores, index = adata.obs_names, columns = [f'Spectra_F{i+1}' for i in range(cell_scores.shape[1])])
    cell_scores.to_csv(f'{outdir}/{prefix}_spectra_scores.txt', sep = '\t', index = True, header = True)
    os.makedirs(f'{outdir}/plots', exist_ok = True)
    for factor in cell_scores.columns:
        if 'X_umap' not in adata.obsm.keys(): continue
        fig = scatterplot_adata(adata, v = cell_scores[factor], rep = 'umap')
        fig.savefig(f'{outdir}/plots/{prefix}_spectra_{factor}_umap.png', bbox_inches = 'tight', dpi = 400)
        plt.close(fig)
    adata.var[['spectra_vocab']].to_csv(f'{outdir}/{prefix}_spectra_vocab.txt', sep = '\t', index = True, header = True)
    
    proj.complete_step('programmes_spectra', dataset, prefix)
    proj.complete_step('programmes_spectra_scores', dataset, prefix)

def run_scired(dataset, prefix, outdir, n_components = 50, n_genes = 2000, 
    covar_cols = [], cell_type = [],           
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
    out_corr = f'{outdir}/{prefix}_scired_correlation.txt'
    out_corr_fig = f'{outdir}/{prefix}_scired_correlation.pdf'
    out_interpretability = f'{outdir}/{prefix}_scired_interpretability.txt'
    out_interpretability_fig = f'{outdir}/{prefix}_scired_interpretability.pdf'
    out_enrichr = f'{outdir}/{prefix}_scired_enrichr.txt'

    import sciRED
    import sciRED.utils
    import statsmodels.api as sm
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

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
        y_varimax_ = pd.read_table(out_scores, index_col = 0)
        loading_varimax_ = pd.read_table(out_loading, index_col = 0)
        y_varimax = y_varimax_.values
        loading_varimax = loading_varimax_.values
        log.log(f'Found existing scIRED output files, loading from {outdir}', calling_file = 'run_scired')
    else:
        # get covariates
        if 'n_umi' not in covar_cols: covar_cols.append('n_umi')
        design_mat = []
        for col in covar_cols:
            if col not in adata.obs.columns:
                raise ValueError(f'Covariate column {col} not found in adata.obs')
            if isinstance(adata.obs[col].dtype, pd.CategoricalDtype) or adata.obs[col].dtype == object:
                dummies = pd.get_dummies(adata.obs[col], prefix = col, drop_first = True)
                design_mat.append(dummies)
            else:
                design_mat.append(adata.obs[[col]])
        design_mat = pd.concat(design_mat, axis = 1)
        design_mat = sm.add_constant(design_mat, has_constant = 'add')

        # factor identification for sciRED
        glm_fit_dict = sciRED.glm.poissonGLM(y, design_mat.astype(float).values)
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
        log.log(f'scIRED gene-level weights saved to {out_loading}', calling_file = 'run_scired')
        log.log(f'scIRED cell-level scores saved to {out_scores}', calling_file = 'run_scired')

    # plot UMAP scatterplot for all factors
    os.makedirs(f'{outdir}/plots', exist_ok = True)
    for factor in tqdm(y_varimax_.columns.tolist(), desc = 'Plotting scIRED cell-level scores in UMAP space'):
        if 'X_umap' not in adata.obsm.keys(): continue
        fig = scatterplot_adata(adata, v = y_varimax_[factor], rep = 'umap')
        fig.savefig(f'{outdir}/plots/{prefix}_scired_{factor}_umap.png', bbox_inches = 'tight', dpi = 400)
        plt.close(fig)
    log.log(f'scIRED UMAP plots saved to {outdir}/plots', calling_file = 'run_scired')

    # pseudotime regression plots
    if len(cell_type) > 0 and 'pseudotime' in adata.obs.columns:
        factor_pseudotime_reg(y_varimax, adata, f'{outdir}/plots/{prefix}_scired_$factor_pseudotime.png', cell_type[0])

    # FCAT analysis for factor importance
    fcat_mat = factor_importance(y_varimax, adata, cell_type, out_fcat, out_fcat_fig)
    log.log(f'scIRED factor importance analysis saved to {out_fcat} and {out_fcat_fig}', calling_file = 'run_scired')

    # correlation analysis
    factor_correlation(loading_varimax_, out_corr, out_corr_fig)

    # enrichment analysis
    enrichr_res = factor_enrichr(loading_varimax_, top_negative = True)
    enrichr_res.to_csv(out_enrichr, sep = '\t', index = False, header = True)
    log.log(f'scIRED factor enrichment analysis saved to {out_enrichr}', calling_file = 'run_scired')

    # correlation with n_umi
    corr_numi = sciRED.utils.corr.get_factor_libsize_correlation(y_varimax, adata.obs['n_umi'].values)
    log.log(f'Max correlation between identified factors and library size (n_umi): {np.abs(corr_numi).max():.4f}', calling_file = 'run_scired')

    # interpretability scoring
    if os.path.isfile(out_interpretability) and not force:
        interpretability_metrics = pd.read_table(out_interpretability, index_col = 0)
        log.log(f'Found existing scIRED interpretability output file, loading from {out_interpretability}', calling_file = 'run_scired')
    else:
        log.log(f'Calculating scIRED factor interpretability metrics', calling_file = 'run_scired')
        from joblib import parallel_backend
        interpretability_metrics = pd.DataFrame(index = [f'F{i+1}' for i in range(y_varimax.shape[1])], columns = [])
        with parallel_backend('threading', n_jobs = 32):
            silhouette_score = sciRED.metrics.kmeans_bimodal_score(y_varimax, time_eff = True)
            bimodality_index = sciRED.metrics.bimodality_index(y_varimax)
        interpretability_metrics['bimodality_score'] = (np.array(silhouette_score) + np.array(bimodality_index)) / 2
        interpretability_metrics['effect_size'] = sciRED.metrics.factor_variance(y_varimax)
        interpretability_metrics['specificity_score'] = sciRED.metrics.simpson_diversity_index(
            fcat_mat.pivot(index = 'explained_factor', columns = 'scired_factor', values = 'fcat_value')
        )
        for col in cell_type:
            if col not in adata.obs.columns or not (isinstance(adata.obs[col].dtype, pd.CategoricalDtype) or adata.obs[col].dtype == object): continue
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
    log.log(f'scIRED factor interpretability analysis saved to {out_interpretability} and {out_interpretability_fig}', calling_file = 'run_scired')

    proj.complete_step('programmes_scired', dataset, prefix)
    proj.complete_step('programmes_scired_scores', dataset, prefix)

def main(args):
    cnmf_outdir = os.path.dirname(os.path.dirname(proj.config['programmes_cnmf'])).replace(
        '$dataset', args.dataset).replace('$prefix', args.prefix) # cNMF automatically creates the $prefix subdirectory
    scired_outdir = os.path.dirname(proj.config['programmes_scired']).replace('$dataset', args.dataset).replace('$prefix', args.prefix)
    spectra_outdir = os.path.dirname(proj.config['programmes_spectra']).replace('$dataset', args.dataset).replace('$prefix', args.prefix)
    if args.cnmf:
        run_cnmf(args.dataset, args.prefix, cnmf_outdir, n_components = args.cnmf_components, 
            cell_type = args.cell_type, force = args.force, worker_id = args.worker)
    if args.scired:
        run_scired(args.dataset, args.prefix, scired_outdir, n_components = args.scired_components,
            n_genes = args.scired_genes, covar_cols = args.scired_covars,
            cell_type = args.cell_type, force = args.force)
    if args.spectra:
        run_spectra(args.dataset, args.prefix, spectra_outdir, cell_type = args.cell_type[0])
        
def add_cmd_args(parser):
    parser.add_argument('--cnmf', action = 'store_true', help = 'Run consensus NMF to identify gene programmes')
    parser.add_argument('--cnmf_components', type = int, nargs = 3, default = (10, 70, 10),
        help = 'Number of components to identify for cNMF (start, stop, step), default: 10 70 10')
    parser.add_argument('--scired', action = 'store_true', help = 'Run scIRED to identify gene programmes')
    parser.add_argument('--scired_components', type = int, default = 50,
        help = 'Number of components to identify for scIRED (default: 50)')
    parser.add_argument('--scired_genes', type = int, default = 2000,
        help = 'Number of highly variable genes to use for scIRED (default: 2000)')
    parser.add_argument('--scired_covars', type = str, nargs = '+', default = ['sex'],
        help = 'Covariate columns in adata.obs to adjust for in scIRED (default: sex)')
    parser.add_argument('--spectra', action = 'store_true', help = 'Run Spectra to identify gene programmes')
    parser.add_argument('--cell_type', type = str, nargs = '+', default = ['Type_updated'],
        help = '''Categorical factors in adata.obs that denote the cell type. 
        Only the first is used for Spectra decomposition and pseudotime regression plots.
        All factors are used for factor importance scoring
        (default: Type_updated)''')
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