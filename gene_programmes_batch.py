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
from gene_programmes import check_cnmf_incomplete, add_cmd_args
add_cmd_args = add_slurm_args_dec(add_cmd_args)

def get_obs_cols(dataset, prefix, default, input_string = 'cell type'):
    import scanpy as sc
    adata = sc.read_h5ad(proj.to_pathname('raw', dataset, prefix), 'r')
    default = [x for x in default if x in adata.obs.columns]
    # take keyboard input to select cell type columns
    log.log(f'Following columns are found in {dataset}/{prefix} metadata:')
    for i, col in enumerate(adata.obs.columns):
        log.log(f'    {i}: {col}')
    selected_cols = input(f'Enter the column numbers for {input_string}, separated by space: \n' + str(default) + ' ').strip()
    selected_cols = [adata.obs.columns[int(x)] for x in selected_cols.split()]
    if len(selected_cols) == 0: selected_cols = default
    if len(selected_cols) == 0: log.warn('No valid cell type column selected/found, please check your input and dataset metadata')
    print()
    log.log('Selected ' + input_string + ' columns: ')
    for col in selected_cols: log.log(f'    {col}')
    return selected_cols

def main(args):
    cnmf_prep_submitter = array_submitter(name = 'cnmf_prep_' + '_'.join(args.datasets),
        partition = 'icelake-himem', n_cpu = 32, timeout = 120)
    cnmf_submitter = array_submitter(name = 'cnmf_batch_' + '_'.join(args.datasets)+'_'+str(max(args.cnmf_components)),
        partition = 'icelake-himem', n_cpu = 4, timeout = 720, dependency = cnmf_prep_submitter)
    scired_submitter = array_submitter(name = 'scired_batch_' + '_'.join(args.datasets),
        partition = 'sapphire', n_cpu = 32, timeout = 720)
    spectra_submitter = array_submitter(name = 'spectra_batch_' + '_'.join(args.datasets),
        partition = 'sapphire', n_cpu = 16, timeout = 720)
    
    h5ad = proj.find_h5ad(args.datasets, normalised = False)
    cnmf_outdir = 'programmes/cnmf/$dataset/$prefix'
    proj.register('programmes_cnmf',f'{cnmf_outdir}/$prefix.spectra.k_$k.dt_$dt.consensus.txt')
    proj.register('programmes_cnmf_scores',f'{cnmf_outdir}/$prefix.usages.k_$k.dt_$dt.consensus.txt')
    scired_outdir = 'programmes/scired/$dataset/$prefix'
    proj.register('programmes_scired',f'{scired_outdir}/$prefix_scired_loadings.txt')
    proj.register('programmes_scired_scores',f'{scired_outdir}/$prefix_scired_scores.txt')
    spectra_outdir = 'programmes/spectra/$dataset/$prefix'
    proj.register('programmes_spectra',f'{spectra_outdir}/$prefix_spectra_loadings.txt')
    proj.register('programmes_spectra_scores',f'{spectra_outdir}/$prefix_spectra_scores.txt')
    cnmf_components_str = ' '.join([str(x) for x in args.cnmf_components])

    for dataset, prefix in h5ad:
        # if cell type is set as default
        if args.cell_type == ['cell_type']: selected_cell_types = get_obs_cols(dataset, prefix, args.cell_type)
        else: selected_cell_types = args.cell_type
        if args.time_key == ['pseudotime', 'age', 'time']: selected_time_keys = get_obs_cols(dataset, prefix, args.time_key, input_string = 'time-related variable')
        else: selected_time_keys = args.time_key

        cmd = f'python gene_programmes.py {dataset} {prefix} --cnmf_components {cnmf_components_str} --cnmf_dt {args.cnmf_dt} '+ \
            f'--scired_components {args.scired_components} --scired_genes {args.scired_genes} --scired_covars {" ".join(args.scired_covars)} '+ \
            f'--cell_type {" ".join(selected_cell_types)} --time_key {" ".join(selected_time_keys)} --embedding {" ".join(args.embedding)}'
        if len(args.projection) > 0: cmd += ' --project ' + ' '.join(args.projection)
        if args.force: cmd += ' --force'
        if args.magma: cmd += f' --magma_out {args.magma_out} --magma'
        if args.cnmf:
            n_cnmf_incomplete = check_cnmf_incomplete(dataset, prefix, 
                range(args.cnmf_components[0], args.cnmf_components[1]+1, args.cnmf_components[2]),
                projection = args.projection)
            n_jobs = max(1, min(100, n_cnmf_incomplete)) # use up to 100 workers
            # if the preprocessing step is not complete, submit a separate job with higher memory just to preprocess files
            norm_counts = os.path.dirname(proj.to_pathname('programmes_cnmf', dataset, prefix, k = 10, dt = '0_1')) + \
                f'/cnmf_tmp/{prefix}.norm_counts.h5ad'.replace('$dataset', dataset).replace('$prefix', prefix)
            log.log(f'Looking for {norm_counts} to determine if cNMF preprocessing is complete')
            if not os.path.isfile(norm_counts):
                cnmf_prep_submitter.add(cmd + ' --cnmf --worker -1')
            for worker_id in range(n_jobs): cnmf_submitter.add(cmd + f' --cnmf --worker {worker_id}')
        if args.scired: scired_submitter.add(cmd + ' --scired')
        if args.spectra: spectra_submitter.add(cmd + ' --spectra')
    cnmf_submitter.submit()
    scired_submitter.submit()
    spectra_submitter.submit()

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