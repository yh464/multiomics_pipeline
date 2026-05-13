#!/usr/bin/env python3
'''
Author: Yuankai He
Correspondence: yh464@cam.ac.uk
Version 1: 2026-05-13

Scripts to read metadata from h5ad files and do simple operations
'''

import os
from _utils.path import project
proj = project()
from _utils.logger import logger
log = logger()
import scanpy as sc

def get_metadata_cols(dataset = None, prefix = None, default = [], input_string = 'cell type', axis = 'obs', adata = None):
    assert axis in ['obs', 'var'], 'axis must be either obs or var'
    if adata is None: adata = sc.read_h5ad(proj.to_pathname('raw', dataset, prefix), 'r')
    df = adata.obs if axis == 'obs' else adata.var
    default = [x for x in default if x in df.columns]
    # take keyboard input to select cell type columns
    log.log('Following columns are found:')
    for i, col in enumerate(df.columns):
        log.log(f'    {i}: {col}')
    selected_cols = input(f'Enter the column numbers for {input_string}, separated by space: \n' + str(default) + ' ').strip()
    selected_cols = [df.columns[int(x)] for x in selected_cols.split()]
    if len(selected_cols) == 0: selected_cols = default
    if len(selected_cols) == 0: log.warn('No valid cell type column selected/found, please check your input and dataset metadata')
    print()
    log.log('Selected ' + input_string + ' columns: ')
    for col in selected_cols: log.log(f'    {col}')
    return selected_cols

def get_embedding_keys(dataset = None, prefix = None, default = [], adata = None):
    if adata is None: adata = sc.read_h5ad(proj.to_pathname('raw', dataset, prefix), 'r')
    default = [x for x in default if x in adata.obsm.keys()]
    log.log('Following embeddings are found:')
    for i, key in enumerate(adata.obsm.keys()):
        log.log(f'    {i}: {key}')
    selected_keys = input(f'Enter the column numbers for embeddings to use, separated by space: \n' + str(default) + ' ').strip()
    selected_keys = [adata.obsm.keys()[int(x)] for x in selected_keys.split()]
    if len(selected_keys) == 0: selected_keys = default
    if len(selected_keys) == 0: log.warn('No valid embedding selected/found, please check your input and dataset obsm')
    print()
    log.log('Selected embeddings: ')
    for key in selected_keys: log.log(f'    {key}')
    return selected_keys

def subset_h5ad(h5ad_in, h5ad_out, gene_subset):
    import scanpy as sc
    if type(gene_subset) == str and os.path.isfile(gene_subset): gene_subset = open(gene_subset).read().splitlines()
    adata = sc.read_h5ad(h5ad_in,'r')
    adata = adata[:, [x for x in gene_subset if x in adata.var_names]].to_memory() # subset to genes in gene_subset that are present in adata
    adata = adata[:, adata.X.sum(axis = 0) > 0] # subset to genes with at least one non-zero count
    sc.write(h5ad_out, adata)
    adata.file.close()