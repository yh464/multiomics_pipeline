#!/usr/bin/env python3
'''
Author: Yuankai He
Correspondence: yh464@cam.ac.uk
Version 1: 2025-12-21

Pipelines to identify gene expression programmes from scRNA-seq data
'''

import os
from _utils.logger import logger
log = logger()
from _utils.path import project
proj = project()

from _utils.slurm import array_submitter, add_slurm_args_dec

def main(args):
    submitter = array_submitter(name = 'gene_programmes_batch_' + '_'.join(args.datasets),
        partition = 'sapphire', n_cpu = 16, timeout = 240)
    
    h5ad = proj.find_h5ad(args.datasets, normalised = False)
    cnmf_outdir = os.path.realpath('../programmes/cnmf/$dataset/$prefix')
    proj.register('programmes_cnmf',f'{cnmf_outdir}/$prefix.usages.k_*.dt_*.consensus.txt')
    scired_outdir = os.path.realpath('../programmes/scired/$dataset/$prefix')
    proj.register('programmes_scired',f'{scired_outdir}/$prefix_scired_loadings.txt')
    proj.register('programmes_scired_scores',f'{scired_outdir}/$prefix_scired_scores.txt')
    cnmf_components_str = ' '.join([str(x) for x in args.cnmf_components])

    for dataset, prefix in h5ad:
        cmd = f'python gene_programmes.py {dataset} {prefix} --cnmf_components {cnmf_components_str} '+ \
            f'--scired_components {args.scired_components} --scired_genes {args.scired_genes} '+ \
            f'--scired_covars {" ".join(args.scired_covars)} --scired_explain {" ".join(args.scired_explain)}'
        if args.force: cmd += ' --force'
        if args.cnmf: submitter.add(cmd + ' --cnmf')
        if args.scired: submitter.add(cmd + ' --scired')
    submitter.submit()

from gene_programmes import add_cmd_args
add_cmd_args = add_slurm_args_dec(add_cmd_args)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'Batch run gene programme identification across multiple datasets')
    parser.add_argument('datasets', nargs = '+', help = 'List of dataset names / prefixes to process')
    parser = add_cmd_args(parser)
    args = parser.parse_args()
    if not args.scired and not args.cnmf:
        log.warn('No method selected, defaults to cNMF')
        args.cnmf = True

    from _utils import cmdhistory
    log.splash(args)
    cmdhistory.log()
    try: main(args)
    except: cmdhistory.errlog()