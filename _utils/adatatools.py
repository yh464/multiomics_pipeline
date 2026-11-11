#!/usr/bin/env python3
'''
Author: Yuankai He
Correspondence: yh464@cam.ac.uk
Version 1: 2026-05-13

Scripts to read metadata from h5ad files and do simple operations
'''

import os, h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from _utils.path import project
proj = project()
from _utils.logger import logger
log = logger()

# accessing data using h5py
def h5py_get_col(col_connection):
    if isinstance(col_connection, h5py._hl.group.Group): # as categorical, contains 'categories' and 'codes'
        col = pd.Series(col_connection['categories'][:][col_connection['codes'][:]].astype(str))
    else:
        col = pd.Series(col_connection[:].astype(str))
    return col

# data check functions
def check_spatial(dataset = None, prefix = None, spatial_key = 'spatial', adata = None):
    if adata is None:
        with h5py.File(proj.to_pathname('raw', dataset, prefix), 'r') as file:
            keys = list(file['obsm'].keys())
    else: keys = adata.obsm.keys()
    if not (out := spatial_key in keys): log.warn(f'Spatial key {spatial_key} not found, please check your input dataset')
    return out

def check_atac(dataset = None, prefix = None, adata = None):
    '''Checks and normalises ATAC peak names, and lifts over between hg19 and hg38 if necessary;
    progress is stored in adata.uns so only need to run once'''

    log.log('Checking if input dataset is a scATAC-seq dataset and if peak names are in the correct format', calling_file = 'check_atac')

    if adata is not None: 
        if 'atac_check_complete' in adata.uns.keys() and adata.uns['atac_check_complete']: return True
        log.warn('check_atac progress will be saved in memory, please save adata to file', calling_file = 'check_atac')
        var_names = adata.var_names
        if var_names.str.contains('ENSG').any():
            log.warn('Gene names in var_names appear to be Ensembl IDs, press Ctrl+C to abort if input dataset is not ATAC-seq', calling_file = 'check_atac')
    else:
        with h5py.File(proj.to_pathname('raw', dataset, prefix), 'r') as file:
            if 'atac_check_complete' in file['uns'].keys() and file['uns']['atac_check_complete'][()]: return True
            var_names = pd.Series(file['var'][file['var'].attrs['_index']][:].astype(str))
            if var_names.str.contains('ENSG').any():
                log.warn('Gene names in var_names appear to be Ensembl IDs, press Ctrl+C to abort if input dataset is not ATAC-seq', calling_file = 'check_atac')

    chrom_col, start_col, end_col = select_metadata_cols(
        dataset = dataset, prefix = prefix, adata = adata,
        axis = 'var', input_string = 'chromosome, start and end, in that order', default = ['Chrom','Start','End'])

    genome_build = input('Enter genome build (hg19 or hg38): (Default: hg38)').strip()
    while genome_build not in ['hg19', 'hg38', '']:
        genome_build = input('Invalid genome build, please enter hg19 or hg38: ').strip()
    genome_build = genome_build or 'hg38'
    lift_to = 'hg38' if genome_build == 'hg19' else 'hg19'

    # load chromosome, start and end coordinates from adata or h5ad file
    coords = dict(hg19 = dict(), hg38 = dict())
    if adata is not None:
        coords[genome_build]['chrom'] = 'chr' + adata.var[chrom_col].astype(str).str.replace('chr', '', regex = False).str.strip().replace(
            {'23': 'X', '24': 'Y', '25': 'M', '26': 'M', 'MT': 'M'})
        coords[genome_build]['start'] = adata.var[start_col].astype(int)
        coords[genome_build]['end'] = adata.var[end_col].astype(int)
    else:
        with h5py.File(proj.to_pathname('raw', dataset, prefix), 'r') as file:
            chrom = h5py_get_col(file['var'][chrom_col])
            coords[genome_build]['chrom'] = 'chr' + pd.Series(chrom).str.replace('chr', '', regex = False).str.strip().replace(
                {'23': 'X', '24': 'Y', '25': 'XY', '26': 'M', 'MT': 'M'})
            coords[genome_build]['start'] = h5py_get_col(file['var'][start_col]).astype(int)
            coords[genome_build]['end'] = h5py_get_col(file['var'][end_col]).astype(int)

    coords[lift_to] = dict(chrom = coords[genome_build]['chrom'], start = [], end = [])

    from liftover import LiftOver
    lo = LiftOver(genome_build, lift_to)
    log.log(f'Lifting over from {genome_build} to {lift_to} using liftover')
    for chrom, start, end in tqdm(zip(coords[genome_build]['chrom'], coords[genome_build]['start'], coords[genome_build]['end']),
        total = len(coords[genome_build]['chrom']), desc = f'Lifting over {len(coords[genome_build]["chrom"])} peaks from {genome_build} to {lift_to}'):
        try:
            new_start = lo.convert_coordinate(chrom, start)
            new_end = lo.convert_coordinate(chrom, end)
            if len(new_start) == 0 or len(new_end) == 0: raise ValueError(f'No lift over result for {chrom}:{start}-{end}')
            if new_start[0][0] != new_end[0][0]: raise ValueError(f'Lift over result for {chrom}:{start}-{end} is on different chromosomes: {new_start[0][0]} vs {new_end[0][0]}')
            coords[lift_to]['start'].append(new_start[0][1])
            coords[lift_to]['end'].append(new_end[0][1])
        except Exception as e:
            log.warn(f'Error lifting over {chrom}:{start}-{end}: {e}', calling_file = 'check_atac')
            coords[lift_to]['start'].append(-1)
            coords[lift_to]['end'].append(-1)

    coords[lift_to]['start'] = pd.Series(coords[lift_to]['start'])
    coords[lift_to]['end'] = pd.Series(coords[lift_to]['end'])

    new_var_names = coords['hg38']['chrom'] + ':' + coords['hg38']['start'].astype(str) + '-' + coords['hg38']['end'].astype(str)

    # store both sets of coordinates and new var_names into adata.var or h5ad file
    # coordinates should be named as 'chrom', 'start_hg19', 'end_hg19', 'start_hg38', 'end_hg38' and moved to the front of adata.var or h5ad file
    # original chromosome, start and end columns should be dropped
    if adata is not None:
        adata.var.drop(columns = [chrom_col, start_col, end_col], inplace = True)
        adata.var.insert(0, 'chrom', coords[genome_build]['chrom'])
        adata.var.insert(1, f'start_{genome_build}', coords[genome_build]['start'])
        adata.var.insert(2, f'end_{genome_build}', coords[genome_build]['end'])
        adata.var.insert(3, f'start_{lift_to}', coords[lift_to]['start'])
        adata.var.insert(4, f'end_{lift_to}', coords[lift_to]['end'])
        adata.var_names = new_var_names
        adata.uns['atac_check_complete'] = True
    else:
        with h5py.File(proj.to_pathname('raw', dataset, prefix), 'r+') as file:
            var = file['var']
            var.move(chrom_col, 'chrom')
            var.move(start_col, f'start_{genome_build}')
            var.move(end_col, f'end_{genome_build}')
            var.create_dataset(f'start_{lift_to}', data = coords[lift_to]['start'].values, dtype = 'int64')
            var.create_dataset(f'end_{lift_to}', data = coords[lift_to]['end'].values, dtype = 'int64')
            var[var.attrs['_index']][:] = new_var_names.values.astype(bytes).astype(object) # bytes stored as objects
            file['uns'].create_dataset('atac_check_complete', data = True, dtype = 'bool')
            var.attrs['column-order'] = np.array(['chrom', f'start_{genome_build}', f'end_{genome_build}', f'start_{lift_to}', f'end_{lift_to}'] + 
                [x for x in var.attrs['column-order'] if x not in [chrom_col, start_col, end_col]]).astype(object) # string as objects

    return True

# functions to prompt to command line to select metadata columns/names
def select_metadata_cols(dataset = None, prefix = None, default = [], input_string = 'cell type', axis = 'obs', adata = None):
    assert axis in ['obs', 'var'], 'axis must be either obs or var'
    if adata is None:
        with h5py.File(proj.to_pathname('raw', dataset, prefix), 'r') as file:
            if 'column-order' in file[axis].attrs.keys(): columns = list(file[axis].attrs['column-order'])
            else: columns = list(file[axis].keys())
    else: columns = adata.obs.columns if axis == 'obs' else adata.var.columns
    default = [x.lower() for x in default]
    default = [x for x in columns if x.lower() in default]
    # take keyboard input to select cell type columns
    log.log('Following columns are found:')
    for i, col in enumerate(columns):
        log.log(f'    {i}: {col}')
    selected_cols = input(f'Enter the column numbers for {input_string}, separated by space: \n' + str(default) + ' ').strip()
    selected_cols = [columns[int(x)] for x in selected_cols.split()]
    if len(selected_cols) == 0: selected_cols = default
    if len(selected_cols) == 0: log.warn('No valid cell type column selected/found, please check your input and dataset metadata')
    print()
    log.log('Selected ' + input_string + ' columns: ')
    for col in selected_cols: log.log(f'    {col}')
    return selected_cols

def select_embedding_keys(dataset = None, prefix = None, default = [], adata = None):
    if adata is None: 
        with h5py.File(proj.to_pathname('raw', dataset, prefix), 'r') as file:
            keys = list(file['obsm'].keys())
    else: keys = adata.obsm.keys()
    default = [x.lower() for x in default]
    default = [x for x in keys if x.lower() in default]
    log.log('Following embeddings are found:')
    for i, key in enumerate(keys):
        log.log(f'    {i}: {key}')
    selected_keys = input(f'Enter the column numbers for embeddings to use, separated by space: \n' + str(default) + ' ').strip()
    selected_keys = [keys[int(x)] for x in selected_keys.split()]
    if len(selected_keys) == 0: selected_keys = default
    if len(selected_keys) == 0: log.warn('No valid embedding selected/found, please check your input and dataset obsm')
    print()
    log.log('Selected embeddings: ')
    for key in selected_keys: log.log(f'    {key}')
    return selected_keys

# functions for file operations
def read_lazy(path, obs = True, var = True, obsm = True):
    import anndata as ad
    adata = ad.read_lazy(path)
    if obs: adata.obs = adata.obs.to_memory()
    if var: adata.var = adata.var.to_memory()
    if obsm: adata.obsm = adata.obsm.to_memory()
    return adata

def subset_h5ad(h5ad_in, h5ad_out, gene_subset):
    import scanpy as sc
    if type(gene_subset) == str and os.path.isfile(gene_subset): gene_subset = open(gene_subset).read().splitlines()
    adata = sc.read_h5ad(h5ad_in,'r')
    adata = adata[:, [x for x in gene_subset if x in adata.var_names]].to_memory() # subset to genes in gene_subset that are present in adata
    adata = adata[:, adata.X.sum(axis = 0) > 0] # subset to genes with at least one non-zero count
    sc.write(h5ad_out, adata)
    adata.file.close()

# functions for in-memory anndata operations
def downsample_adata(adata, downsample = -1, group_by = None, seed = 19260817):
    # if downsample is -1, downsample so that the total number of cells is around 100000
    # no downsampling if fewer than 150000 cells
    if downsample == -1 and adata.shape[0] > 150000:
        if group_by: downsample = max(1, int(100000 / adata.obs.groupby(group_by).ngroups))
        else: downsample = 100000
    if downsample > 0 and group_by:
        log.log('Number of cells per cell type:', calling_file = 'downsample_adata')
        for cell_type_name in adata.obs.groupby(group_by).size().index:
            log.log(f'  {cell_type_name}: {adata.obs[adata.obs[group_by] == cell_type_name].shape[0]}', calling_file = 'downsample_adata')
        selected_cells = adata.obs.groupby(group_by).apply(lambda x: x.sample(n = min(downsample, x.shape[0]), replace = False, random_state = seed))
        if isinstance(selected_cells.index, pd.MultiIndex): selected_cells.index = selected_cells.index.droplevel(0)
    elif downsample > 0:
        selected_cells = adata.obs.sample(n = min(downsample, adata.shape[0]), replace = False, random_state = seed)
    else: selected_cells = adata.obs
    adata = adata[selected_cells.index, :].to_memory()
    return adata
