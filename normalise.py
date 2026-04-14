#!/usr/bin/env python3
'''
Author: Yuankai He
Correspondence: yh464@cam.ac.uk
Version 1: 2026-03-12

Pipelines to normalise the scRNA-seq data and preprocess for scdrs
'''

import os
from _utils.logger import logger
log = logger()
from _utils.path import project
proj = project()

@log.profile
def main(args):
    import scanpy as sc
    import scdrs
    import pandas as pd

    ref = pd.read_table('/rds/project/rds-Nl99R8pHODQ/toolbox/magma/ENSG.gene.loc', header = None).drop_duplicates(5).set_index(5)
    ref.columns = ['ENSG','CHR','START','STOP','GENE','DIRE']
    h5ad_raw = proj.to_pathname('raw', args.dataset, args.prefix)
    h5ad_norm = proj.to_pathname('normalised', args.dataset, args.prefix)
    h5ad_scdrs = proj.to_pathname('scdrs', args.dataset, args.prefix)
    adata_raw = sc.read_h5ad(h5ad_raw, 'r')
    if adata_raw.var_names.isin(ref['ENSG']).mean() > 0.9:
        log.log('Most genes in the dataset are in the MAGMA reference, normalising and subsetting to these genes for scDRS')
        separate_ensg_only = False
    else: separate_ensg_only = True

    if not os.path.isfile(h5ad_norm) or args.force:
        adata_norm = adata_raw[:,adata_raw.var_names.isin(ref['ENSG'])].to_memory() if not separate_ensg_only else adata_raw.to_memory()
        sc.pp.filter_cells(adata_norm, min_genes = 250)
        sc.pp.filter_genes(adata_norm, min_cells = 50)
        sc.pp.normalize_total(adata_norm, target_sum = 1e6)
        sc.pp.log1p(adata_norm)
        if not 'X_pca' in adata_norm.obsm: sc.pp.pca(adata_norm, n_comps = 50)
        sc.pp.neighbors(adata_norm, n_neighbors = 15, n_pcs = 20)
        if not 'X_umap' in adata_norm.obsm and args.umap: sc.tl.umap(adata_norm)
        if not 'X_tsne' in adata_norm.obsm and args.tsne: sc.tl.tsne(adata_norm, n_pcs = 50)
        scdrs.preprocess(adata_norm)
        sc.write(h5ad_norm, adata_norm)

    if not separate_ensg_only:
        if args.force and os.path.islink(h5ad_scdrs):
            os.remove(h5ad_scdrs)
        if not os.path.isfile(h5ad_scdrs):
            os.symlink(h5ad_norm, h5ad_scdrs)

    if not separate_ensg_only: return # already completed at this point, no need to preprocess separately for scDRS
    
    adata_ensg = adata_raw[:,adata_raw.var_names.isin(ref['ENSG'])].to_memory()
    sc.pp.filter_cells(adata_ensg, min_genes = 250)
    sc.pp.filter_genes(adata_ensg, min_cells = 50)
    sc.pp.normalize_total(adata_ensg, target_sum = 1e6)
    sc.pp.log1p(adata_ensg)
    if not 'X_pca' in adata_ensg.obsm: sc.pp.pca(adata_ensg, n_comps = 50)
    sc.pp.neighbors(adata_ensg, n_neighbors = 15, n_pcs = 20)

    # use dimensionality reduction from the full gene set
    adata_norm = sc.read_h5ad(h5ad_norm, 'r')
    if args.umap and 'X_umap' in adata_norm.obsm:
        adata_ensg.obsm['X_umap'] = adata_norm.obsm['X_umap']
    if args.tsne and 'X_tsne' in adata_norm.obsm:
        adata_ensg.obsm['X_tsne'] = adata_norm.obsm['X_tsne']
    
    scdrs.preprocess(adata_ensg)
    sc.write(h5ad_scdrs, adata_ensg)

def add_cmd_args(parser):
    parser.add_argument('--scdrs', action = 'store_true', help = 'Preprocess for scDRS (extract only genes in MAGMA gene set)')
    parser.add_argument('--umap', action = 'store_true', help = 'Calculate UMAP coordinates if not present')
    parser.add_argument('--tsne', action = 'store_true', help = 'Calculate tSNE coordinates if not present')
    parser.add_argument('-f','--force', action = 'store_true', help = 'Force overwrite')
    return parser

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'Log-normalise scRNA-seq data and options to preprocess for scDRS')
    parser.add_argument('dataset', type = str, help = 'Dataset name')
    parser.add_argument('prefix', type = str, help = 'Prefix for output files')
    parser = add_cmd_args(parser)
    args = parser.parse_args()
    log.splash(args)
    main(args)