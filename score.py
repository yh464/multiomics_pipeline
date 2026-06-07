#!/usr/bin/env python3
'''
Author: Yuankai He
Correspondence: yh464@cam.ac.uk
2025-09-01

A python implementation of cepo to save computing time

Requires following inputs: 
    h5ad file and columns for cell classifications
'''

import gc, os
from scipy import sparse
from time import perf_counter as t
from tqdm import tqdm
import pandas as pd
import numpy as np
from _utils.logger import logger
log = logger()
from _utils.path import project
proj = project()

def segidx(df, n):
    nz = df['nz']/n
    ms = df['s']/n; ms[ms==0] = np.nan
    sds = ((df['sumsq']/n - ms**2) * n/(n-1)) ** 0.5
    cvs = pd.Series(sds/ms)
    x1 = nz.rank()/(len(nz)+1)
    x2 = 1 - cvs.rank()/(len(cvs)+1)
    return (x1 + x2)/2

def generate_tempfile(adata, subset, block_size = 50000):
    if adata[subset,:].shape[0] > block_size:
        from time import perf_counter as t
        tic = t()
        df = pd.DataFrame(index = adata.var.index, columns = ['s','sumsq','nz'], data = 0)
        cells = adata[subset,:].obs_names.to_numpy()
        for i in range(0, adata[subset,:].shape[0], block_size):
            log.log(f'Processing block {i}-{min(i+block_size, adata[subset,:].shape[0])} of {adata[subset,:].shape[0]}, time = {t()-tic:.2f}s')
            temp = generate_tempfile(adata, cells[i:min(i+block_size, adata[subset,:].shape[0])], block_size = block_size)
            df += temp; del temp
        return df
    mat = adata[subset,:].to_memory().X
    mat = sparse.csr_array(mat)
    s = mat.sum(axis = 0)
    mat.data **= 2
    ssq = mat.sum(axis = 0)
    mat.data **= 0.5
    nz = (mat > 0).sum(axis = 0)
    del mat
    gc.collect()
    return pd.DataFrame(dict(s = s, sumsq = ssq, nz = nz), index = adata.var.index)

def singlebatchcepo(adata, genes, celltypes, mincells = 20, exprspct:float = 0.05, tempdir = '/home/yh464/rds/hpc-work/temp/'):
    # adata is backed
    # columns = genes, rows = cells

    os.makedirs(tempdir, exist_ok = True)
    celltypes = pd.Series(celltypes).astype('category')
    celltypes_df = pd.get_dummies(celltypes)
    for ct in celltypes_df.columns:
        if celltypes_df[ct].sum() < mincells:
            celltypes_df = celltypes_df.drop(columns = ct)
    if celltypes_df.shape[1] == 0: return []

    # first generate temporary files
    for ct in tqdm(celltypes_df.columns, desc = 'Generating temporary files', total = celltypes_df.shape[1]):
        tempfile = f'{tempdir}/cepo.{ct}.txt'
        if not os.path.isfile(tempfile):
            generate_tempfile(adata, celltypes_df[ct].values).to_csv(tempfile, sep = '\t', index = True, header = True)

    # filter genes such that each gene is expressed in at least exprspct of cells in at least one cell type
    keep = pd.DataFrame(index = genes, columns = celltypes_df.columns, data = False)
    for ct in celltypes_df.columns:
        tempfile = f'{tempdir}/cepo.{ct}.txt'
        temp = pd.read_table(tempfile, index_col = 0).loc[:,'nz'] >= celltypes_df[ct].sum() * exprspct
        temp.index = genes
        keep[ct] = temp
    keep = keep.any(axis = 1).values
    log.log(f'Kept {keep.sum()}/{len(keep)} genes')

    segmat = pd.DataFrame(index = genes[keep], columns = celltypes_df.columns, data = float(0))
    for ct in celltypes_df.columns:
        tempfile = f'{tempdir}/cepo.{ct}.txt'
        temp = pd.read_table(tempfile, index_col = 0)
        temp.index = genes
        segmat[ct] = segidx(temp.loc[keep, :], celltypes_df[ct].sum())
    
    out = pd.DataFrame(index = genes, columns = celltypes_df.columns, data = float(0))
    for ct in celltypes_df.columns:
        out.loc[keep,ct] = segmat[ct] - segmat.drop(ct,axis=1).mean(axis = 1)
    out = out.fillna(0)
    return out

def run_cepo(args):
    import anndata
    out_file = f'{args.out}/{args.prefix}.cepo.txt' if args.prefix.find(args.dataset) > -1 \
        else f'{args.out}/{args.dataset}_{args.prefix}.cepo.txt'
    if os.path.isfile(out_file) and not args.force: return
    tic = t()
    adata = anndata.io.read_h5ad(args._in, 'r')
    genes = adata.var.index
    log.log(f'Read in data: {adata.shape} in {t()-tic:.2f}s')

    out_dfs = []
    for label in args.label:
        tempdir = f'/home/yh464/rds/hpc-work/temp/cepo/{os.path.basename(args._in)[:-5]}/{label}'
        tempfile = f'{tempdir}/cepo.{label}.txt'
        if os.path.isfile(tempfile) and not args.force:
            df = pd.read_table(tempfile, index_col = 0)
            out_dfs.append(df)
            log.log(label); continue
        if label not in adata.obs.columns: Warning(f'Label {label} not found in adata.obs, skipping'); continue
        df = singlebatchcepo(adata, genes, adata.obs[label], mincells = 20, exprspct = 0.05, tempdir = tempdir)
        gc.collect()
        df.columns = [f'cepo.{label}.{x}' for x in df.columns]
        out_dfs.append(df)
        df.to_csv(tempfile, sep = '\t', index = True, header = True)
        del df
        log.log(f'Computed {label} in {t()-tic:.2f}s')
    out_dfs = pd.concat(out_dfs, axis = 1)
    adata.file.close()
    out_dfs.columns = out_dfs.columns.str.replace(' ','_').str.replace('/','_').str.replace('-','_')
    out_dfs.index.name = 'gene'
    out_dfs.to_csv(out_file, sep = '\t', index = True, header = True)
    return out_dfs

def main(args):
    if args.cepo: return run_cepo(args)
    else: raise NotImplementedError('Only cepo score calculation is implemented in this script for now, please use --cepo flag')

def add_cmd_args(parser):
    parser.add_argument('-o', '--out', help = 'Output diretory', default = '../gene_score')
    parser.add_argument('-c', '--cell_type', nargs = '*', help = 'Columns containing cell classifications/annotations in the h5ad dataset', default = [])
    parser.add_argument('--cepo', action = 'store_true', help = 'Calculate cepo scores')
    parser.add_argument('-f','--force', help = 'Force overwrite', action = 'store_true', default = False)
    return parser

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'Log-normalise scRNA-seq data and options to preprocess for scDRS')
    parser.add_argument('dataset', type = str, help = 'Dataset name')
    parser.add_argument('prefix', type = str, help = 'Prefix for output files')
    parser = add_cmd_args(parser)
    args = parser.parse_args()

    import os
    args._in = os.path.realpath(args._in); args.out = os.path.realpath(args.out)

    from _utils import logger, cmdhistory
    logger.splash(args)
    cmdhistory.log()
    try: main(args)
    except: cmdhistory.errlog()