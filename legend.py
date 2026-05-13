#!/usr/bin/env python3
'''
Author: Yuankai He
Correspondence: yh464@cam.ac.uk
Version 1: 2026-05-13

A quick utility to generate cell type legends for scatterplots
'''

import os
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from _plots.colourcode_scatterplot import scatterplot_adata
from _utils.path import project
proj = project()
from _utils.logger import logger
log = logger()
from _utils.adatatools import get_metadata_cols, get_embedding_keys
from tqdm import tqdm

def main(args):
    h5ad = proj.find_h5ad(args.datasets, normalised = False)
    for dataset, prefix in h5ad:
        h5ad_raw = proj.to_pathname('raw', dataset, prefix)
        adata = sc.read_h5ad(h5ad_raw, 'r')
        args.cell_type = [x for x in args.cell_type if x in adata.obs.columns]
        args.embedding = [x for x in args.embedding if x in adata.obsm.keys()]
        if args.cell_type in [['cell_type'], []]: selected_cell_types = get_metadata_cols(adata = adata, default = args.cell_type)
        else: selected_cell_types = args.cell_type
        if all([x in adata.obsm.keys() for x in args.embedding]):
            selected_embeddings = args.embedding
        else: selected_embeddings = get_embedding_keys(adata = adata, default = args.embedding)
        for cell_type_col in tqdm(selected_cell_types, desc = f'Processing {dataset}/{prefix}'):
            out_fig = f'{args.out}/{dataset}_{prefix}_{cell_type_col}_legend.png' if prefix.find(dataset) == -1 else \
                f'{args.out}/{prefix}_{cell_type_col}_legend.png'
            if not args.force and os.path.exists(out_fig): continue
            scatterplot_adata(adata, v = adata.obs[cell_type_col], rep = selected_embeddings[0])
            plt.savefig(out_fig, dpi = 400, bbox_inches = 'tight')
            plt.close()
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'A quick utility to generate cell type legends for scatterplots')
    parser.add_argument('datasets', nargs = '+', help = 'Datasets to process')
    parser.add_argument('--cell_type', nargs = '+', default = ['cell_type'], 
        help = 'Cell type column(s) in adata.obs to use for legend. Leave as default to be prompted to select from available columns.')
    parser.add_argument('--embedding', nargs = '+', default = ['X_umap'], 
        help = 'Embedding(s) in adata.obsm to use for scatterplot coordinates. Default: X_umap. If not available, you will be prompted to select.')
    parser.add_argument('--out', default = '../legends', help = 'Output directory for legends')
    parser.add_argument('-f', '--force', action = 'store_true', help = 'Force overwrite')
    args = parser.parse_args()
    log.splash(args)
    main(args)