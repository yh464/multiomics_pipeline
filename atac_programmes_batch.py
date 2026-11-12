#!/usr/bin/env python3
'''
Author: Yuankai He
Correspondence: yh464@cam.ac.uk
Version 1: 2026-11-04

Pipelines to identify gene programmes from scATAC-seq data
'''

import os
from _utils.logger import logger
log = logger()
from _utils.path import project
proj = project()

from _utils.slurm import array_submitter, add_slurm_args_dec
from atac_programmes import add_cmd_args, check_cistopic_complete, check_scopen_complete
add_cmd_args = add_slurm_args_dec(add_cmd_args)
from _utils.adatatools import select_metadata_cols, check_atac

def main(args):
    cistopic_prep_submitter = array_submitter(name = 'cistopic_prep_batch_' + '_'.join(args.datasets),
        partition = 'icelake-himem', n_cpu = 32, timeout = 720, env = '/rds/project/rds-Nl99R8pHODQ/toolbox/pycisTopic')
    cistopic_submitter = array_submitter(name = 'cistopic_batch_' + '_'.join(args.datasets)+'_'+str(max(args.cistopic_components)),
        partition = 'icelake-himem', n_cpu = 16, timeout = 720, env = '/rds/project/rds-Nl99R8pHODQ/toolbox/pycisTopic', dependency = cistopic_prep_submitter)
    scopen_prep_submitter = array_submitter(name = 'scopen_prep_batch_' + '_'.join(args.datasets),
        partition = 'icelake-himem', n_cpu = 32, timeout = 720, env = '/rds/project/rds-Nl99R8pHODQ/toolbox/scOpen')
    scopen_submitter = array_submitter(name = 'scopen_batch_' + '_'.join(args.datasets)+'_'+str(max(args.scopen_components)),
        partition = 'icelake-himem', n_cpu = 16, timeout = 720, env = '/rds/project/rds-Nl99R8pHODQ/toolbox/scOpen', dependency = scopen_prep_submitter)
    
    h5ad = proj.find_h5ad(args.datasets, normalised = False)
    cistopic_outdir = 'programmes/cistopic/$dataset/$prefix'
    proj.register('atac_programmes_cistopic',f'{cistopic_outdir}/$prefix.cistopic.k_$k.topic_loadings.txt.gz')
    scopen_outdir = 'programmes/scopen/$dataset/$prefix'
    proj.register('atac_programmes_scopen',f'{scopen_outdir}/$prefix.scopen.k_$k.dt_$dt.consensus.txt')

    for dataset, prefix in h5ad:
        if not check_atac(dataset, prefix): continue
        
        # if cell type is set as default
        if args.cell_type == ['cell_type']: selected_cell_types = select_metadata_cols(dataset, prefix, args.cell_type)
        else: selected_cell_types = args.cell_type
        if args.time_key == ['pseudotime', 'age', 'time']: selected_time_keys = select_metadata_cols(dataset, prefix, args.time_key, input_string = 'time-related variable')
        else: selected_time_keys = args.time_key

        flags = []
        for key in ['cistopic_components', 'cistopic_alpha', 'cistopic_eta', 'mallet', 'scopen_components', 'downsample', 'niter']:
            flags.append(f'--{key}')
            if type(getattr(args, key)) in [list, range, tuple]: flags.extend([str(x) for x in getattr(args, key)])
            else: flags.append(str(getattr(args, key)))
        flags.extend(['--cell_type'] + selected_cell_types + ['--time_key'] + selected_time_keys + ['--embedding'] + args.embedding)
        if getattr(args, 'force'): flags.append('--force')
        if getattr(args, 'magma'): flags.extend(['--magma', '--magma_out', args.magma_out])

        cmd = f'python atac_programmes.py {dataset} {prefix} '+ ' '.join(flags)

        if args.cistopic: 
            cmds = []
            for i in (incomplete := check_cistopic_complete(dataset, prefix, 
                n_components = range(args.cistopic_components[0], args.cistopic_components[1]+1, args.cistopic_components[2])
                ) if len(args.cistopic_components) == 3 else [args.cistopic_components]):
                cmds.append(cmd + f' --cistopic --worker {i}')
            if len(incomplete) == 0:
                cmds.append(cmd + f' --cistopic --worker 0')

            cistopic_dir = os.path.dirname(proj.to_pathname('atac_programmes_cistopic', dataset = dataset, prefix = prefix, k = incomplete[0]))
            if not os.path.isfile(f'{cistopic_dir}/temp/corpus.mallet'):
                log.log(f'Pre-processed corpus file not found for {dataset} {prefix}, submitting pre-processing job')
                cistopic_prep_submitter.add(cmds.pop(0))
                for c in cmds: cistopic_submitter.add(c)
            else:
                for c in cmds: cistopic_submitter.add(c)

        if args.scopen:
            cmds = []
            for i in (incomplete := check_scopen_complete(dataset, prefix, 
                n_components = range(args.scopen_components[0], args.scopen_components[1]+1, args.scopen_components[2])
                ) if len(args.scopen_components) == 3 else [args.scopen_components]):
                cmds.append(cmd + f' --scopen --worker {i}')
            if len(incomplete) == 0:
                cmds.append(cmd + f' --scopen --worker 0')

            scopen_dir = os.path.dirname(proj.to_pathname('atac_programmes_scopen', dataset = dataset, prefix = prefix, k = incomplete[0]))
            if not os.path.isfile(f'{scopen_dir}/temp/{prefix}.tfidf.h5ad'):
                log.log(f'Pre-processed TF-IDF file not found for {dataset} {prefix}, submitting pre-processing job')
                scopen_prep_submitter.add(cmds.pop(0))
                for c in cmds: scopen_submitter.add(c)
            else:
                for c in cmds: scopen_submitter.add(c)

    cistopic_submitter.submit()
    scopen_submitter.submit()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'Batch run ATAC-seq programme identification across multiple datasets')
    parser.add_argument('datasets', nargs = '+', help = 'List of dataset names / prefixes to process')
    parser = add_cmd_args(parser)
    args = parser.parse_args()
    if not args.cistopic and not args.scopen:
        log.warn('No method selected, defaults to cistopic')
        args.cistopic = True

    from _utils import cmdhistory
    log.splash(args)
    cmdhistory.log()
    try: main(args)
    except: cmdhistory.errlog()