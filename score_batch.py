#!/usr/bin/env python3
'''
Author: Yuankai He
Correspondence: yh464@cam.ac.uk
2025-08-28

Generates gene scores for MAGMA gene covariate analysis

Requires following inputs: 
    directory containing single-cell h5ad files
    columns containing cell classifications
'''

import os
from _utils.logger import logger
log = logger()
from _utils.path import project
proj = project()

def main(args):
    from _utils.slurm import array_submitter
    from _utils.adatatools import get_metadata_cols
    import scanpy as sc
    submitter = array_submitter(name = 'sc_score', n_cpu = 16, timeout = 480)

    h5ad = proj.find_h5ad(args.datasets, normalised = False)
    for dataset, prefix in h5ad:
        out_file = f'{args.out}/{prefix}.cepo.txt'
        if os.path.isfile(out_file) and not args.force: continue
        h5 = proj.to_pathname('raw', dataset, prefix)
        adata = sc.read_h5ad(h5, 'r')
        cell_type = [col for col in args.cell_type if col in adata.obs.columns]
        if cell_type is []: cell_type = get_metadata_cols(dataset, prefix, input_string = 'cell type', adata = adata)
        # cmd = ['Rscript', 'sc_score.r','-i', h5, '-o', f'{args.out}/{prefix}', '--label'] + args.label
        cmd = ['python', 'sc_score_cepo.py',dataset, prefix,'-o', args.out, '--label'] + cell_type
        if args.cepo: cmd.append('--cepo')
        if args.force: cmd.append('-f')
        submitter.add(' '.join(cmd))
    submitter.submit()
    return submitter

if __name__ == '__main__':
    from score import add_cmd_args
    from _utils.slurm import slurm_parser
    parser = slurm_parser(description = 'This script scores genes for cell type specificity for MAGMA gene covariate analysis')
    parser.add_argument('datasets', nargs = '+', help = 'Datasets to process, format <dataset>/<prefix>')
    parser = add_cmd_args(parser)
    args = parser.parse_args()
    args.out = os.path.realpath(args.out)
    if not args.cepo: log.warn('No scoring method specified, defaulting to cepo'); args.cepo = True

    from _utils import logger, cmdhistory
    logger.splash(args)
    cmdhistory.log()
    try: main(args)
    except: cmdhistory.errlog()