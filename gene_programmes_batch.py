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
from gene_programmes import check_cnmf_completed

def main(args):
    cnmf_submitter = array_submitter(name = 'cnmf_batch_' + '_'.join(args.datasets)+'_'+str(args.cnmf_components),
        partition = 'icelake-himem', n_cpu = 4, timeout = 120)
    scired_submitter = array_submitter(name = 'scired_batch_' + '_'.join(args.datasets),
        partition = 'sapphire', n_cpu = 32, timeout = 720)
    spectra_submitter = array_submitter(name = 'spectra_batch_' + '_'.join(args.datasets),
        partition = 'sapphire', n_cpu = 16, timeout = 720)
    
    h5ad = proj.find_h5ad(args.datasets, normalised = False)
    cnmf_outdir = os.path.realpath('../programmes/cnmf/$dataset/$prefix')
    proj.register('programmes_cnmf',f'{cnmf_outdir}/$prefix.gene_spectra_score.k_*.dt_*.consensus.txt')
    proj.register('programmes_cnmf_scores',f'{cnmf_outdir}/$prefix.usages.k_*.dt_*.consensus.txt')
    scired_outdir = os.path.realpath('../programmes/scired/$dataset/$prefix')
    proj.register('programmes_scired',f'{scired_outdir}/$prefix_scired_loadings.txt')
    proj.register('programmes_scired_scores',f'{scired_outdir}/$prefix_scired_scores.txt')
    spectra_outdir = os.path.realpath('../programmes/spectra/$dataset/$prefix')
    proj.register('programmes_spectra',f'{spectra_outdir}/$prefix_spectra_loadings.txt')
    proj.register('programmes_spectra_scores',f'{spectra_outdir}/$prefix_spectra_scores.txt')
    cnmf_components_str = ' '.join([str(x) for x in args.cnmf_components])

    for dataset, prefix in h5ad:
        cmd = f'python gene_programmes.py {dataset} {prefix} --cnmf_components {cnmf_components_str} --cnmf_dt {args.cnmf_dt} '+ \
            f'--scired_components {args.scired_components} --scired_genes {args.scired_genes} '+ \
            f'--scired_covars {" ".join(args.scired_covars)} --cell_type {" ".join(args.cell_type)}'
        if args.force: cmd += ' --force'
        if args.cnmf:
            h5ad_raw = proj.to_pathname('raw', dataset, prefix)
            outdir = os.path.realpath(cnmf_outdir).replace('$dataset', dataset).replace('$prefix', prefix)
            import cnmf
            cnmf_obj = cnmf.cNMF(output_dir = outdir, name = prefix)
            if not os.path.isfile(cnmf_obj.paths['normalized_counts']) or not os.path.isfile(cnmf_obj.paths['tpm']):
                log.log(f'Preparing cNMF normalised count files for {dataset}/{prefix}')
                cnmf_obj.prepare(counts_fn = h5ad_raw, 
                    components = range(args.cnmf_components[0], args.cnmf_components[1]+1, args.cnmf_components[2]), 
                    n_iter = 100, seed = 19260817)
            else: log.log(f'cNMF normalised count file for {dataset}/{prefix} found at {cnmf_obj.paths["normalized_counts"]}')
            cnmf_complete = check_cnmf_completed(dataset, prefix, cnmf_outdir, 
                range(args.cnmf_components[0], args.cnmf_components[1]+1, args.cnmf_components[2]))
            n_jobs = 1 if cnmf_complete else 100
            for worker_id in range(n_jobs): cnmf_submitter.add(cmd + f' --cnmf --worker {worker_id}')
        if args.scired: scired_submitter.add(cmd + ' --scired')
        if args.spectra: spectra_submitter.add(cmd + ' --spectra')
    cnmf_submitter.submit()
    scired_submitter.submit()
    spectra_submitter.submit()

from gene_programmes import add_cmd_args
add_cmd_args = add_slurm_args_dec(add_cmd_args)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'Batch run gene programme identification across multiple datasets')
    parser.add_argument('datasets', nargs = '+', help = 'List of dataset names / prefixes to process')
    parser = add_cmd_args(parser)
    args = parser.parse_args()
    if not args.scired and not args.cnmf and not args.spectra:
        log.warn('No method selected, defaults to cNMF')
        args.cnmf = True

    from _utils import cmdhistory
    log.splash(args)
    cmdhistory.log()
    try: main(args)
    except: cmdhistory.errlog()