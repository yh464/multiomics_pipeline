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
from _utils.slurm import array_submitter, add_slurm_args_dec
from normalise import add_cmd_args

def main(args):
    submitter = array_submitter(name = 'normalise_batch_' + '_'.join(args.datasets),
        partition = 'icelake-himem', n_cpu = 16, timeout = 720)
    # no need to register 'normalised' as that is specified in _utils.path.project as a default path
    scdrs_outdir = 'scdrs/$dataset/$prefix'
    proj.register('scdrs', f'{scdrs_outdir}/$prefix.h5ad')
    
    h5ad = proj.find_h5ad(args.datasets, normalised = False)
    for dataset, prefix in h5ad:
        cmd = f'python normalise.py {dataset} {prefix}'
        if args.scdrs: cmd += ' --scdrs'
        if args.force: cmd += ' --force'
        if args.umap: cmd += ' --umap'
        if args.tsne: cmd += ' --tsne'
        submitter.add(cmd)
    submitter.submit()

add_cmd_args = add_slurm_args_dec(add_cmd_args)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'Log-normalise scRNA-seq data and options to preprocess for scDRS')
    parser.add_argument('datasets', nargs = '+', type = str, help = 'Dataset names / prefixes to process')
    parser = add_cmd_args(parser)
    args = parser.parse_args()
    if not args.umap and not args.tsne: 
        log.warn('No dimensionality reduction method specified, defaulting to UMAP')
        args.umap = True # default to UMAP if no dimensionality reduction specified

    log.splash(args)
    from _utils import cmdhistory
    cmdhistory.log()
    try: main(args)
    except: cmdhistory.errlog()