import os
from _utils.logger import logger
log = logger()
from _utils.path import project
proj = project()
from _utils.adatatools import check_spatial

def main(args):
    from _utils.slurm import array_submitter
    out_dir = '/rds/project/rds-Nl99R8pHODQ/multiomics/gsmap'
    h5ad_raw = proj.find_h5ad(args.datasets, normalised = False)

    submitter = array_submitter(n_cpu = 40, timeout = 720, name = 'preprocess_sc_4gsmap',
        wd = os.path.dirname(os.path.realpath(__file__)))

    for dataset, prefix in h5ad_raw:
        if not check_spatial(dataset, prefix): continue
        out_prefix = f'{dataset}_{prefix}' if prefix.find(dataset) == -1 else prefix
        check_file = f'{out_dir}/{out_prefix}/generate_ldscore/{out_prefix}_generate_ldscore.done'
        if os.path.isfile(check_file): 
            log.log(f'Preprocessing for {dataset}/{prefix} is already completed')
            continue
        input_h5ad = proj.to_pathname('raw', dataset, prefix)
        cmd = f'bash preprocess_sc_4gsmap.sh {input_h5ad} {out_prefix}'
        submitter.add(cmd)
    submitter.submit()

if __name__ == '__main__':
    from _utils.slurm import slurm_parser
    parser = slurm_parser(description = 'Preprocess single-cell datasets for gsMap')
    parser.add_argument('datasets', nargs = '+', required = True, help = 'Datasets to preprocess, format <dataset>/<prefix>')
    args = parser.parse_args()
    main(args)