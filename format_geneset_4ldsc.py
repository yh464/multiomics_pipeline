#!/usr/bin/env python3
'''
Author: Yuankai He
Correspondence: yh464@cam.ac.uk
Version 1: 2026-09-16

A quick utility to process gene sets for stratified LDSC analysis
'''

import os, glob
import pandas as pd
from _utils.path import project
proj = project()
from _utils.logger import logger
log = logger()
from tqdm import tqdm

def main(args):
    root_dir = proj.project_root
    gene_set = f'{root_dir}/gene_set'
    gene_score = f'{root_dir}/gene_score'
    out_dir = f'{root_dir}/snp_annot'
    gsets = []; gscores = []

    # read gene location file
    geneloc = pd.read_table(args.geneloc, header = None, names = ['ENSG','CHR','FROM','TO','LABEL','DIR']).drop_duplicates('ENSG').set_index('ENSG')
    geneloc['FROM'] -= args.window[0]*1000; geneloc['TO'] += args.window[-1]*1000
    geneloc = geneloc.loc[~geneloc['CHR'].isin(['X','Y','MT']), :] # autosomes only
    geneloc['CHR'] = geneloc['CHR'].astype(int)
    geneloc = geneloc.sort_values(['CHR', 'FROM'])

    # find files for each dataset
    for dataset in args.datasets:
        gset = glob.glob(f'{gene_set}/*{dataset}*.txt')
        gscore = glob.glob(f'{gene_score}/*{dataset}*.txt')
        if len(gset) == 0 and len(gscore) == 0:
            log.warn(f'No gene set or gene score files found for dataset {dataset}, skipping'); continue
        gsets += [(x, x.replace(f'{gene_set}/', '')[:-4]) for x in gset]; gscores += [(x, x.replace(f'{gene_score}/', '')[:-4]) for x in gscore]

    log.log(f'Found {len(gsets)} gene set datasets and {len(gscores)} gene score datasets for processing')
    for _, prefix in gsets: log.log(f'    {prefix} (gene set)')
    for _, prefix in gscores: log.log(f'    {prefix} (gene score)')

    if args.gnova and len(gscores) > 0:
        log.warn('GNOVA analysis is only supported for gene sets, skipping following files')
        for _, prefix in gscores: log.warn(f'    {prefix}')

    if args.chrom == 'all':
        # running batch mode
        from _utils.slurm import array_submitter
        submitter = array_submitter(name = 'format_geneset_4ldsc', partition = 'icelake-himem', n_cpu = 4, timeout = 120)
        for _, prefix in gsets:
            for chrom in range(1, 23):
                submitter.add(f'python {__file__} {prefix} --chrom {chrom} --window {" ".join([str(x) for x in args.window])} '+
                    f'--geneloc {args.geneloc} --ref {args.ref} {"--gnova" if args.gnova else ""} {"--force" if args.force else ""}')
        for _, prefix in gscores:
            for chrom in range(1, 23):
                submitter.add(f'python {__file__} {prefix} --chrom {chrom} --window {" ".join([str(x) for x in args.window])} '+
                    f'--geneloc {args.geneloc} --ref {args.ref} {"--force" if args.force else ""}') # removes the --gnova option for gene scores
        submitter.submit()
        return
    
    else:
        chrom = int(args.chrom)
        if len(gsets) + len(gscores) > 1: log.warn('Multiple datasets found, check if names are duplicated; files may be overwritten')

        ref_chr = pd.read_table(args.ref.replace('%chrom', str(chrom)), header = None, usecols = [0,1,3], names = ['CHR','SNP','BP'])
        ref_chr = ref_chr.loc[:, ['CHR','BP','SNP']]
        ref_chr['CM'] = 0; ref_chr['base'] = 1
        log.log(f'Read {ref_chr.shape[0]} SNPs from reference file for chromosome {chrom}')
        geneloc_chr = geneloc.loc[geneloc['CHR'] == chrom, ['FROM','TO']]
        log.log(f'Found {geneloc_chr.shape[0]} genes on chromosome {chrom} in the gene location file')

        # process gene sets
        for gset, prefix in gsets:
            os.makedirs(f'{out_dir}/{prefix}', exist_ok = True)
            out_file = f'{out_dir}/{prefix}/%chrom.annot'
            out_gnova = f'{out_dir}/{prefix}/%chrom.gnova'
            if os.path.isfile(out_file.replace('%chrom', str(chrom))) and not args.force and (
                os.path.isfile(out_gnova.replace('%chrom', str(chrom))) or not args.gnova):
                log.log(f'Annotation file for dataset {prefix} on chromosome {chrom} already exists, skipping'); continue

            gset_data = open(gset).read().splitlines()
            gset_names = [prefix + '_' + x.split()[0] for x in gset_data]

            gset_matrix = pd.DataFrame(0, index = geneloc_chr.index, columns = gset_names)
            for i, line in enumerate(gset_data):
                gset_matrix.loc[geneloc_chr.index.isin(line.split()[1:]), gset_names[i]] = 1
            gset_matrix = gset_matrix.loc[gset_matrix.sum(axis = 1) > 0, :]

            out_chr = pd.concat([ref_chr, pd.DataFrame(0, index = ref_chr.index, columns = gset_names)], axis = 1)
            for g, row in tqdm(gset_matrix.iterrows(), total = gset_matrix.shape[0]):
                start = geneloc_chr.loc[g, 'FROM']; end = geneloc_chr.loc[g, 'TO']
                out_chr.loc[out_chr['BP'].between(start, end), row.index[row == 1]] = 1
            out_chr.to_csv(out_file.replace('%chrom', str(chrom)), sep = '\t', index = False)

            if args.gnova:
                tmp = out_chr.iloc[:, 5:]
                tmp.loc[tmp.sum(axis = 1) == 0, 'no_annotation'] = 1
                tmp.to_csv(out_gnova.replace('%chrom', str(chrom)), sep = '\t', index = False)
            log.log(f'Saved gene set annotations for dataset {prefix} on chromosome {chrom}')

        # process gene scores
        for gscore, prefix in gscores:
            os.makedirs(f'{out_dir}/{prefix}', exist_ok = True)
            out_file = f'{out_dir}/{prefix}/%chrom.annot'
            if os.path.isfile(out_file.replace('%chrom', str(chrom))) and not args.force:
                log.log(f'Annotation file for dataset {prefix} on chromosome {chrom} already exists, skipping'); continue

            gscore_data = pd.read_table(gscore, header = None, index_col = 0)
            gscore_data = gscore_data.loc[gscore_data.index.isin(geneloc_chr.index), :]

            out_chr = pd.concat([ref_chr, pd.DataFrame(0, index = ref_chr.index, columns = gscore_data.columns)], axis = 1)
            for g, row in tqdm(gscore_data.iterrows(), total = gscore_data.shape[0]):
                start = geneloc_chr.loc[g, 'FROM']; end = geneloc_chr.loc[g, 'TO']
                out_chr.loc[out_chr['BP'].between(start, end), gscore_data.columns] += row['SCORE']
            out_chr.to_csv(out_file.replace('%chrom', str(chrom)), sep = '\t', index = False)
            log.log(f'Saved gene score annotations for dataset {prefix} on chromosome {chrom}')


if __name__ == '__main__':
    from _utils.slurm import slurm_parser
    parser = slurm_parser(description = 'A quick utility to process gene sets for stratified LDSC analysis')
    parser.add_argument('datasets', nargs = '+', help = 'Gene sets')
    parser.add_argument('--chrom', default = 'all', help = 'Chromosome to process (NB manually specifying this option will trigger interactive run)')
    parser.add_argument('--window', nargs = '+', default = [10, 10], type = int, help = 'Window size (kb) for gene set analysis, enter one value for symmetric windows or two values for up/downstream')
    parser.add_argument('--geneloc', default = '/rds/project/rds-Nl99R8pHODQ/toolbox/magma/ENSG.gene.loc', help = 'Gene location file for MAGMA')
    parser.add_argument('--ref', default = '/rds/project/rds-Nl99R8pHODQ/ref/1000g_eur_ldsc/chr%chrom.bim', help = 'Reference SNP list for LDSC')
    parser.add_argument('--gnova', action = 'store_true', help = 'Format gene sets for GNOVA analysis')
    parser.add_argument('-f', '--force', action = 'store_true', help = 'Force overwrite')
    args = parser.parse_args()
    if len(args.window) not in [1,2]: log.error('Window size must be 1-2 integer values')
    log.splash(args)
    main(args)