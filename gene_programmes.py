#!/usr/bin/env python3
'''
Author: Yuankai He
Correspondence: yh464@cam.ac.uk
Version 1: 2025-12-21

Pipelines to identify gene expression programmes from scRNA-seq data
'''

# scanpy takes long to import so needs to be imported in each sub-function
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

def subset_h5ad(h5ad_in, h5ad_out, gene_subset):
    import scanpy as sc
    if type(gene_subset) == str and os.path.isfile(gene_subset): gene_subset = open(gene_subset).read().splitlines()
    adata = sc.read_h5ad(h5ad_in,'r')
    adata = adata[:, [x for x in gene_subset if x in adata.var_names]]
    sc.write(h5ad_out, adata)
    adata.close()

def factor_enrichr(scores, top_negative = True, top = [50, 100, 200, 300, 500]):
    from _utils.enrichr import enrichr_continuous
    out = []
    for col in scores.columns:
        for t in top:
            log.log(f'Enrichr analysis for factor {col} on top {t} genes', calling_file = 'factor_enrichr')
            enrichr_res = enrichr_continuous(scores, by = col, top = t, top_negative = top_negative, 
                use_background = False, silent = True)
            enrichr_res.insert(0, 'factor', col)
            enrichr_res['top'] = t
            out.append(enrichr_res)
    out = pd.concat(out, axis = 0)
    return out

def factor_importance(scores, adata, factors_to_explain, out_tabular, out_fig):
    # FCAT analysis for factor importance
    import sciRED.ensembleFCA
    if isinstance(scores, pd.DataFrame): scores = scores.values # FCAT function accepts numpy array as input
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
    for i in corr_mat.index: corr_mat.loc[i,i] = 0
    log.log(f'Highest positive correlation between factors: {corr_mat.values.max():.4f}', calling_file = 'factor_correlation')
    log.log(f'Highest negative correlation between factors: {corr_mat.values.min():.4f}', calling_file = 'factor_correlation')
    for i in corr_mat.index: corr_mat.loc[i,i] = 1
    corr_mat.to_csv(out_tabular, sep = '\t', index = True, header = True)
    fig = corr_heatmap_wide_format(corr_mat)
    fig.savefig(out_fig, bbox_inches = 'tight')
    plt.close(fig)
    return corr_mat

def factor_embedding(scores, adata, out_fig, embedding_key = ['X_umap'], force = False):
    log.log(f'Plotting cell-level scores in embedding space: ' + ', '.join(embedding_key), calling_file = 'run_cnmf')
    os.makedirs(os.path.dirname(out_fig), exist_ok = True)
    if not out_fig.endswith('.png') and not out_fig.endswith('.pdf'):
        out_fig += '.png'
    scores.columns = [f'F{i+1}' for i in range(scores.shape[1])]
    score_cols = scores.columns.tolist()
    for emb_key in embedding_key:
        if not emb_key in adata.obsm.keys():
            log.warn(f'Embedding key {emb_key} not found in adata.obsm, skipping.', calling_file = 'factor_embedding', warning = True)
            continue
        scores = pd.concat([scores, adata.obs], axis = 1)
        for col in tqdm(score_cols, desc = 'Plotting factor embedding regression'):
            figname = out_fig.replace('$factor', col).replace('$embedding', emb_key.replace('X_', ''))
            if os.path.isfile(figname) and not force: continue
            fig = scatterplot_adata(adata, v = scores[col], rep = emb_key)
            fig.savefig(figname, bbox_inches = 'tight', dpi = 400)
            plt.close(fig)
    log.log(f'Factor embedding plots saved to {os.path.dirname(out_fig)}', calling_file = 'factor_embedding')

def factor_time_reg(scores, adata, out_fig, cell_type_key, time_keys = ['pseudotime'], force = False):
    log.log(f'Plotting factor temporal regression for: ' + ', '.join(time_keys), calling_file = 'run_cnmf')
    os.makedirs(os.path.dirname(out_fig), exist_ok = True)
    if not out_fig.endswith('.png') and not out_fig.endswith('.pdf'):
        out_fig += '.png'
    scores.columns = [f'F{i+1}' for i in range(scores.shape[1])]
    score_cols = scores.columns.tolist()
    for time_key in time_keys:
        time_xlabel = '' if time_key.contains('pseudo') else time_key.replace('_',' ')
        scores = pd.concat([scores, adata.obs[[cell_type_key, time_key]]], axis = 1).dropna()
        for col in tqdm(score_cols, desc = 'Plotting factor temporal regression'):
            fig_1order = out_fig.replace('$factor', col).replace('$timekey', time_key)
            fig_2order = out_fig.replace('$factor', col).replace('$timekey', time_key).replace('.png', '_order2.png')
            if os.path.isfile(fig_1order) and os.path.isfile(fig_2order) and not force: continue
            fig = temporal_regplot(scores, x = time_key, y = col, hue = cell_type_key, xlabel = time_xlabel)
            fig.savefig(out_fig.replace('$factor', col).replace('$timekey', time_key), bbox_inches = 'tight', dpi = 400)
            plt.close(fig)
            fig = temporal_regplot(scores, x = time_key, y = col, hue = cell_type_key, order = 2, xlabel = time_xlabel)
            fig.savefig(out_fig.replace('$factor', col).replace('$timekey', time_key).replace('.png', '_order2.png'), bbox_inches = 'tight', dpi = 400)
            plt.close(fig)
    log.log(f'Factor temporal regression plots saved to {os.path.dirname(out_fig)}', calling_file = 'run_cnmf')

def run_cnmf(dataset, prefix, outdir, n_components = range(5, 41, 1), density_threshold = 0.1, 
    cell_type = [], embedding = ['X_umap'], time_keys = ['pseudotime'],
    seed = 19260817, force = False, worker_id = 0, n_iter = 100, savedir = None, projection = []):
    '''
    run consensus NMF on input h5ad file (RAW COUNTS)
    outdir: output directory
    n_components: number of components to identify
    random_state: random seed
    '''
    import scanpy as sc
    h5ad_raw = proj.to_pathname('raw', dataset, prefix)
    if len(projection) > 0:
        projection_dataset = projection[0][0]
        projection_prefix = projection[0][1]
        new_prefix = f'{prefix}_hvg_{projection_dataset}_{projection_prefix}' if projection_dataset not in projection_prefix else f'{prefix}_hvg_{projection_prefix}'
        tmpdir = f'{proj.project_root}/temp'
        os.makedirs(tmpdir, exist_ok = True)
        projected_h5ad = f'{tmpdir}/{dataset}_{new_prefix}.h5ad'
        projection_genes = os.path.dirname(proj.to_pathname('programmes_cnmf', projection_dataset, projection_prefix)) + f'/{projection_prefix}.overdispersed_genes.txt'
        if worker_id == 0 and not os.path.isfile(projected_h5ad):
            subset_h5ad(h5ad_raw, projected_h5ad, projection_genes)
        elif not os.path.isfile(projected_h5ad) or not os.access(projected_h5ad, os.R_OK):
            log.log(f'Waiting for projected h5ad file to be generated at {projected_h5ad}', calling_file = 'run_cnmf')
            time.sleep(60)
        log.log(f'Re-fitting cNMF using highly variable genes from {projection_dataset}/{projection_prefix}', calling_file = 'run_cnmf')
        prefix = new_prefix
        h5ad_raw = projected_h5ad
    outdir = os.path.realpath(outdir).replace('$dataset', dataset).replace('$prefix', prefix)

    import cnmf
    log.log(f'Conducting cNMF on {h5ad_raw}', calling_file = 'run_cnmf')
    log.log(f'Output directory: {outdir}/{prefix}', calling_file = 'run_cnmf')
    cnmf_obj = cnmf.cNMF(output_dir = outdir, name = prefix)

    # check progress and preprocess data
    tic = time.perf_counter()
    while not os.path.isfile(cnmf_obj.paths['normalized_counts']) or not os.path.isfile(cnmf_obj.paths['tpm']) or \
        not os.access(cnmf_obj.paths['normalized_counts'], os.R_OK) or not os.access(cnmf_obj.paths['tpm'], os.R_OK):
        if worker_id == 0: # prevent other workers from simultaneously writing files
            cnmf_obj.prepare(counts_fn = h5ad_raw, components = n_components, n_iter = n_iter, seed = seed)
        else: 
            time.sleep(1)
            if time.perf_counter() - tic > 600:
                log.warn('Waiting too long for normalised count file to be generated', calling_file = 'run_cnmf')
                return
    if worker_id == 0: 
        log.log('Setting cNMF runtime parameters', calling_file = 'run_cnmf')
        replicate_params, run_params = cnmf_obj.get_nmf_iter_params(
            ks = n_components, n_iter = n_iter, random_state_seed = seed,
            beta_loss = 'frobenius', init = 'random', alpha_usage = 0.0, alpha_spectra = 0.0, max_iter = 1000
        )
        cnmf_obj.save_nmf_iter_params(replicate_params, run_params)
        cnmf_obj.update_nmf_iter_params()
        log.log('Saved cNMF runtime parameters', calling_file = 'run_cnmf')
    else: time.sleep(5) # this step is fast enough, no time limit needed
    
    # run NMF iterations
    cnmf_obj.factorize(worker_i = worker_id, total_workers = 100, skip_completed_runs = (not force))
    if not check_cnmf_completed(dataset, prefix, n_components = n_components, n_iter = n_iter):
        log.warn('Waiting for other workers to complete iterations', calling_file = 'run_cnmf')
        return
    
    # FOLLOWING ANALYSES ARE ONLY CONDUCTED AFTER ALL WORKERS HAVE COMPLETED NMF ITERATIONS
    # combine iterations
    for k in n_components:
        if os.path.isfile(cnmf_obj.paths['merged_spectra'] % k) and not force: continue
        cnmf_obj.combine_nmf(k)
    if len(n_components) > 1:
        # PCA scree plot
        adata = sc.read_h5ad(cnmf_obj.paths['normalized_counts'], 'r')
        if not 'pca' in adata.uns.keys() or adata.uns['pca']['variance_ratio'].size < max(n_components):
            adata = adata.to_memory()
            sc.pp.pca(adata, n_comps = max(n_components)+10)
            sc.write(cnmf_obj.paths['normalized_counts'].replace('.h5ad','.pca.h5ad'), adata) # prevent old file from being overwritten
            os.remove(cnmf_obj.paths['normalized_counts'])
            os.rename(cnmf_obj.paths['normalized_counts'].replace('.h5ad','.pca.h5ad'), cnmf_obj.paths['normalized_counts'])
        total_variance_explained = np.cumsum(adata.uns['pca']['variance_ratio'])
        fig, ax = plt.subplots(figsize = (5, 3))
        ax.plot(np.arange(1, len(adata.uns['pca']['variance_ratio'])+1), adata.uns['pca']['variance_ratio'], color = 'k')
        ax1 = ax.twinx()
        ax1.plot(np.arange(1, len(adata.uns['pca']['variance_ratio'])+1), total_variance_explained, color = 'r')
        ax.set_title('PCA Scree Plot')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Variance Explained')
        fig.savefig(f'{outdir}/{prefix}/{prefix}_cnmf_k{total_variance_explained.size}_screeplot.pdf', bbox_inches = 'tight')
        plt.close(fig)
        log.log(f'PCA scree plot saved to {outdir}/{prefix}/{prefix}_cnmf_k{total_variance_explained.size}_screeplot.pdf', calling_file = 'run_cnmf')
        log.log(f'{(adata.uns["pca"]["variance_ratio"] > 0.05).sum()} PCs explain more than 5% variance', calling_file = 'run_cnmf')
        log.log(f'{(adata.uns["pca"]["variance_ratio"] > 0.01).sum()} PCs explain more than 1% variance', calling_file = 'run_cnmf')
        log.log('First 10 PCs:', calling_file = 'run_cnmf')
        for i in range(10):
            log.log(f'    PC{i+1}: {adata.uns["pca"]["variance_ratio"][i]*100:.2f}%', calling_file = 'run_cnmf')

        # error/stability plot
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
    log.log(f'Conducting consensus factor decomposition with k = {k_optim} and density threshold = {density_threshold}', calling_file = 'run_cnmf')
    density_threshold_str = str(density_threshold).replace('.','_')
    if not os.path.isfile(cnmf_obj.paths['consensus_spectra__txt'].replace(r'%d', str(k_optim)).replace(r'%s', density_threshold_str)) or force:
        cnmf_obj.consensus(k = k_optim, density_threshold = density_threshold)

    def cnmf_downstream(usages, loadings, adata, cell_type, outdir, prefix, k_optim, force, savedir = None):
        loadings = loadings.T

        plot_dir = f'{outdir}/{prefix}/k_{k_optim}_plots'
        os.makedirs(plot_dir, exist_ok = True)

        # UMAP plot of cell-level scores
        rep_names = [x for x in adata.obsm.keys() if x in ['X_umap', 'X_tsne'] + embedding]
        if len(rep_names) > 0: factor_embedding(usages, adata, f'{plot_dir}/{prefix}_cnmf_k{k_optim}_$factor_$embedding.png', 
            embedding_key = rep_names, force = force)
        else: log.warn('No embedding found in adata.obsm, skipping cNMF factor embedding plots.', calling_file = 'run_cnmf', warning = True)

        # pseudotime regression plots
        adata.obs.columns = adata.obs.columns.str.lower()
        time_columns = list(set(['age','time','pseudotime'] + time_keys))
        if len(cell_type) > 0 and adata.obs.columns.intersection(time_columns).size > 0:
            factor_time_reg(usages, adata, f'{plot_dir}/{prefix}_cnmf_k{k_optim}_$factor_$timekey.png', 
                cell_type[0], time_keys = adata.obs.columns.intersection(time_columns), force = force)
        else: log.log('No time-related columns found in adata.obs, skipping cNMF factor pseudotime regression plots.', calling_file = 'run_cnmf')
        
        # factor importance scoring
        out_fcat = f'{outdir}/{prefix}/{prefix}_cnmf_k{k_optim}_fcat.txt'
        out_fcat_fig = f'{outdir}/{prefix}/{prefix}_cnmf_k{k_optim}_fcat.pdf'
        if (not os.path.isfile(out_fcat) or force) and len(cell_type) > 0 and cell_type[0] in adata.obs.columns:
            log.log('Conducting cNMF factor importance analysis', calling_file = 'run_cnmf')
            factor_importance(usages.values, adata, cell_type, out_fcat, out_fcat_fig)
        log.log(f'cNMF factor importance analysis saved to {out_fcat} and {out_fcat_fig}', calling_file = 'run_cnmf')

        # correlation and enrichment analysis
        out_corr = f'{outdir}/{prefix}/{prefix}_cnmf_k{k_optim}_correlation.txt'
        out_corr_fig = f'{outdir}/{prefix}/{prefix}_cnmf_k{k_optim}_correlation.pdf'
        out_enrichr = f'{outdir}/{prefix}/{prefix}_cnmf_k{k_optim}_enrichr.txt'
        if not os.path.isfile(out_corr) or force: factor_correlation(loadings, out_corr, out_corr_fig)
        log.log(f'cNMF factor correlation plot saved to {out_corr_fig}', calling_file = 'run_cnmf')
        if not os.path.isfile(out_enrichr) or force:
            enrichr_res = factor_enrichr(loadings, top_negative = False)
            enrichr_res.to_csv(out_enrichr, sep = '\t', index = False, header = True)
        log.log(f'cNMF factor enrichment analysis saved to {out_enrichr}', calling_file = 'run_cnmf')

        if savedir is not None:
            os.makedirs(savedir, exist_ok = True)
            loadings.index.name = 'gene'
            loadings.columns = [f'{prefix}.cnmf_k{k_optim}.F{i+1}' for i in range(loadings.shape[1])]
            loadings.to_csv(f'{savedir}/{prefix}.cnmf_k{k_optim}.txt', sep = '\t', index = True, header = True)
            log.log(f'Formatted cNMF loadings for MAGMA GSEA analysis saved to {savedir}/{prefix}.cnmf_k{k_optim}.txt', calling_file = 'run_cnmf')
    
    usages = pd.read_table(cnmf_obj.paths['consensus_usages__txt'].replace(r'%d', str(k_optim)).replace(r'%s', density_threshold_str), index_col = 0)
    loadings = pd.read_table(cnmf_obj.paths['consensus_spectra__txt'].replace(r'%d', str(k_optim)).replace(r'%s', density_threshold_str), index_col = 0)
    adata = sc.read_h5ad(h5ad_raw, 'r')
    cnmf_downstream(usages, loadings, adata, cell_type, outdir, prefix, k_optim, force, savedir)

    # enforce projection using available loadings from projection_dataset/projection_prefix
    if len(projection) > 0:
        projection_loadings = proj.to_pathname('programmes_cnmf', projection_dataset, projection_prefix, k = k_optim, dt = density_threshold_str)
        if not os.path.isfile(projection_loadings):
            log.warn(f'Projection loadings from {projection_dataset}/{projection_prefix} not found, skipping projection step.', calling_file = 'run_cnmf')
        else:
            from sklearn.decomposition import NMF
            log.log(f'Projecting input data onto loadings from {projection_dataset}/{projection_prefix}', calling_file = 'run_cnmf')
            projected_prefix = prefix.replace('_hvg_','_proj_')
            projected_usages_file = proj.to_pathname('programmes_cnmf_scores', dataset, projected_prefix, k = k_optim, dt = density_threshold_str)
            os.makedirs(os.path.dirname(projected_usages_file), exist_ok = True)
            if not os.path.isfle(projected_usages_file) or force:
                adata_normalised = sc.read_h5ad(cnmf_obj.paths['normalized_counts'])
                projection_loadings = pd.read_table(projection_loadings, index_col = 0) # after transpose, columns = genes, index = factors
                projection_loadings = projection_loadings.loc[:, projection_loadings.columns.intersection(adata_normalised.var_names)]
                nmf = NMF(n_components = projection_loadings.shape[0], init = 'random', random_state = seed, max_iter = 1000)
                nmf.components_ = projection_loadings.values
                projected_usages = nmf.transform(adata_normalised.X)
                projected_usages = pd.DataFrame(projected_usages, index = adata_normalised.obs_names, 
                    columns = [i+1 for i in range(projected_usages.shape[1])])
                projected_usages.to_csv(projected_usages_file, sep = '\t', index = True, header = True)
            else:
                projected_usages = pd.read_table(projected_usages_file, index_col = 0)
            cnmf_downstream(projected_usages, projection_loadings, adata, cell_type, outdir, projected_prefix, projection_loadings.shape[0], force, savedir = None)
        
        return # do not register on the progress file if it is a projection

    proj.complete_step('programmes_cnmf', dataset, prefix)
    proj.complete_step('programmes_cnmf_scores', dataset, prefix)

def check_cnmf_completed(dataset, prefix, n_components = range(5, 41, 1), n_iter = 100, projection = []):
    '''check if cNMF has been completed for given dataset / prefix'''
    proj = project() # need to re-initialise project as this function is called in gene_programmes_batch
    projection = proj.find_h5ad(projection, long = True)
    if len(projection) > 0:
        projection_dataset = projection[0][0]
        projection_prefix = projection[0][1]
        prefix = f'{prefix}_hvg_{projection_dataset}_{projection_prefix}' if projection_dataset not in projection_prefix else f'{prefix}_hvg_{projection_prefix}'
    outdir = os.path.dirname(proj.to_pathname('programmes_cnmf', dataset, prefix))
    n_spectra_complete = 0
    if not os.path.isdir(f'{outdir}/cnmf_tmp'): os.makedirs(f'{outdir}/cnmf_tmp')
    for f in os.listdir(f'{outdir}/cnmf_tmp'):
        for k in n_components:
            if fnmatch(f, f'{prefix}.spectra.k_{k}.iter_*.df.npz'): n_spectra_complete += 1
    if n_spectra_complete < len(n_components)*n_iter:
        log.log(f'{n_spectra_complete} / {len(n_components)*n_iter} cNMF iterations completed for {dataset}/{prefix}', calling_file = 'check_cnmf_completed')
        return False
    log.log(f'All {len(n_components)*n_iter} cNMF iterations completed for {dataset}/{prefix}', calling_file = 'check_cnmf_completed')
    return True

def run_spectra(dataset, prefix, outdir, cell_type):
    '''
    run Spectra on input h5ad file (RAW COUNTS)
    outdir: output directory
    n_components will be estimated from data
    '''
    import scanpy as sc    
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
    covar_cols = [], cell_type = [], embedding = ['X_umap'], time_keys = ['pseudotime'],         
    seed = 19260817, force = False, savedir = None):
    '''
    run consensus NMF on input h5ad file
    h5ad: input h5ad file path or AnnData object, RAW COUNTS
    outdir: output directory
    n_components: number of components to identify
    random_state: random seed
    '''
    import scanpy as sc
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
    plot_dir = f'{outdir}/plots'; os.makedirs(plot_dir, exist_ok = True)
    rep_names = [x for x in adata.obsm.keys() if x in ['X_umap', 'X_tsne'] + embedding]
    if len(rep_names) > 0: factor_embedding(y_varimax_, adata, f'{plot_dir}/{prefix}_scired_$factor_$embedding.png', 
        embedding_key = rep_names, force = force)
    else: log.warn('No embedding found in adata.obsm, skipping scIRED factor embedding plots.', calling_file = 'run_scired', warning = True)

    # pseudotime regression plots
    adata.obs.columns = adata.obs.columns.str.lower()
    time_columns = list(set(['age','time','pseudotime'] + time_keys))
    if len(cell_type) > 0 and adata.obs.columns.intersection(time_columns).size > 0:
        factor_time_reg(y_varimax_, adata, f'{plot_dir}/{prefix}_scired_$factor_$timekey.png', cell_type[0], force = force)
    else: log.log('No time-related columns found in adata.obs, skipping scIRED factor pseudotime regression plots.', calling_file = 'run_scired')

    # FCAT analysis for factor importance
    if not os.path.isfile(out_fcat) or not os.path.isfile(out_fcat_fig) or force:
        fcat_mat = factor_importance(y_varimax_, adata, cell_type, out_fcat, out_fcat_fig)
    log.log(f'scIRED factor importance analysis saved to {out_fcat} and {out_fcat_fig}', calling_file = 'run_scired')

    # correlation analysis
    factor_correlation(loading_varimax_, out_corr, out_corr_fig)

    # enrichment analysis
    if not os.path.isfile(out_enrichr) or force:
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

    if savedir is not None:
        os.makedirs(savedir, exist_ok = True)
        loading_varimax_.index.name = 'gene'
        loading_varimax_.columns = [f'{prefix}.scired.F{i+1}' for i in range(loading_varimax_.shape[1])]
        loading_varimax_.to_csv(f'{savedir}/{prefix}.scired.txt', sep = '\t', index = True, header = True)

@log.profile
def main(args):
    projection = proj.find_h5ad(args.projection, long = True)
    if len(projection) > 1: 
        projection = []
        log.warn('Multiple datasets found for projection, ignoring --project argument')
    cnmf_outdir = os.path.dirname(os.path.dirname(proj.to_pathname('programmes_cnmf', args.dataset, args.prefix))) # cNMF automatically creates the $prefix subdirectory
    scired_outdir = os.path.dirname(os.path.dirname(proj.to_pathname('programmes_scired', args.dataset, args.prefix)))
    spectra_outdir = os.path.dirname(os.path.dirname(proj.to_pathname('programmes_spectra', args.dataset, args.prefix)))
    if args.cnmf:
        run_cnmf(args.dataset, args.prefix, cnmf_outdir, n_components = args.cnmf_components, density_threshold = args.cnmf_dt,
            cell_type = args.cell_type, embedding = args.embedding, time_keys = args.time_keys, seed = 19260817,
            force = args.force, worker_id = args.worker, savedir = args.magma_out if args.magma else None, projection = projection)
    if args.scired:
        run_scired(args.dataset, args.prefix, scired_outdir, n_components = args.scired_components,
            n_genes = args.scired_genes, covar_cols = args.scired_covars, cell_type = args.cell_type, embedding = args.embedding, time_keys = args.time_keys, 
            force = args.force, savedir = args.magma_out if args.magma else None, projection = projection)
    if args.spectra:
        run_spectra(args.dataset, args.prefix, spectra_outdir, cell_type = args.cell_type[0], savedir = args.magma_out if args.magma else None, embedding = args.embedding, time_keys = args.time_keys)
        
def add_cmd_args(parser):
    parser.add_argument('--cnmf', action = 'store_true', help = 'Run consensus NMF to identify gene programmes')
    parser.add_argument('--cnmf_components', type = int, nargs = 3, default = (5, 40, 1),
        help = 'Number of components to identify for cNMF (start, stop, step), default: 5 40 1')
    parser.add_argument('--cnmf_dt', type = float, default = 0.1,
        help = 'Density threshold for cNMF consensus spectra (default: 0.1)')
    parser.add_argument('--scired', action = 'store_true', help = 'Run scIRED to identify gene programmes')
    parser.add_argument('--scired_components', type = int, default = 50,
        help = 'Number of components to identify for scIRED (default: 50)')
    parser.add_argument('--scired_genes', type = int, default = 2000,
        help = 'Number of highly variable genes to use for scIRED (default: 2000)')
    parser.add_argument('--scired_covars', type = str, nargs = '+', default = ['sex'],
        help = 'Covariate columns in adata.obs to adjust for in scIRED (default: sex)')
    parser.add_argument('--spectra', action = 'store_true', help = 'Run Spectra to identify gene programmes')

    parser.add_argument('--project', type = str, nargs = '*', default = [], dest = 'projection',
        help = 'Use pre-computed gene programmes of another dataset and project onto the current dataset. Format <dataset>/<prefix>')
    
    parser.add_argument('--cell_type', type = str, nargs = '+', default = ['Type_updated'],
        help = '''Categorical factors in adata.obs that denote the cell type. 
        Only the first is used for Spectra decomposition and pseudotime regression plots.
        All factors are used for factor importance scoring
        (default: Type_updated)''')
    parser.add_argument('--embedding', type = str, nargs = '*', default = ['X_umap', 'X_tsne'],
        help = 'Embedding representations in adata.obsm to use for factor embedding plots (default: X_umap X_tsne)')
    parser.add_argument('--time_keys', type = str, nargs = '*', default = ['pseudotime', 'age', 'time'],
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
    args.cnmf_components = range(args.cnmf_components[0], args.cnmf_components[1]+1, args.cnmf_components[2])
    log.splash(args)
    main(args)