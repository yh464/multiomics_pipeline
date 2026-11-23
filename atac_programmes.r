#### Information ####
# R components of scATAC-seq biological programme extraction pipeline
# Author: Yuankai He (yh464@cam.ac.uk)
# V1.0:   2026-11-22

#### Reading command line input ####
library(argparse)
library(here)
parser = ArgumentParser(description = 'R pipelines to identify gene programmes from scATAC-seq data: cicero, chromVAR')
parser$add_argument('-i','--in', dest = 'input', help = 'Input H5AD file path', required = TRUE)
parser$add_argument('-o','--out', dest = 'out', help = 'Output prefix', required = TRUE)
parser$add_argument('--cicero', action = 'store_true', help = 'Run Cicero to identify cis co-accessible networks (CCANs)')
parser$add_argument('--chromvar', action = 'store_true', help = 'Run chromVAR to identify transcription factor (TF) programmes')

parser$add_argument('--downsample', type = 'integer', default = -1, 
    help = 'Downsample the input AnnData object to a maximum number of cells, default: -1 (automatic), specify 0 to disable')
parser$add_argument('--cell_type', nargs = '*', default = c('cell_type'), help = 'Cell type column name in the input AnnData object, default: cell_type')
parser$add_argument('-f','--force', action = 'store_true', help = 'Force overwrite')

args = parser$parse_args(commandArgs(TRUE))
args$input = normalizePath(args$input)
args$out = normalizePath(args$out, mustWork = FALSE)
if (args$cicero & args$chromvar) {stop('Please choose either --cicero or --chromvar, not both.')}
print('Input arguments:')
print(args)

library(BiocParallel)
# get n_cpu from system variable SLURM_CPUS_PER_TASK
n_cpu = as.numeric(Sys.getenv('SLURM_CPUS_PER_TASK', unset = 1))
# use detectCores as a fallback if SLURM_CPUS_PER_TASK is not set
if (is.na(n_cpu) || n_cpu < 1) n_cpu = parallel::detectCores()
register(MulticoreParam(n_cpu))

#### Utility to downsample adata ####
downsample_adata = function(adata, downsample, cell_type = NULL, seed = 19260817) {
    if (downsample == -1 & nrow(adata$obs) > 150000) {
        if (!is.null(cell_type)) {
            downsample = max(1, floor(100000 / length(unique(adata$obs[[cell_type]]))))
        } else {
            downsample = 100000
        }
    }

    if (downsample > 0 & !is.null(cell_type)) {
        message('Number of cells per cell type:')
        selected_cells = adata$obs %>% group_by(.data[[cell_type]]) %>% group_modify(~ {
            n_cells = nrow(.x)
            message('  ', unique(.x[[cell_type]]), ': ', n_cells)
            n_sample = min(downsample, n_cells)
            .x[sample(n_cells, n_sample, replace = FALSE), ]
        }) %>% ungroup()
    } else if (downsample > 0) {
        n_cells = nrow(adata$obs)
        message('Total number of cells: ', n_cells)
        n_sample = min(downsample, n_cells)
        selected_cells = adata$obs[sample(n_cells, n_sample, replace = FALSE), ]
    } else {
        selected_cells = adata$obs
    }

    adata = adata[selected_cells$index, ] %>% as_InMemoryAnnData() # read HDF5AnnData into memory
    adata = adata[, Matrix::rowSums(adata$X) > 0]
    adata = adata[Matrix::colSums(adata$X) > 0, ]
    return(adata)
}

#### Cicero main code ####
run.cicero = function(args) { # run_cicero is a cicero package function so use dot
    library(anndataR)
    library(Matrix)
    library(monocle3)
    library(cicero)
    library(tidyverse)

    if (length(args$cell_type) > 0) ct = args$cell_type[1] else ct = NULL
    adata = read_h5ad(args$input, as = 'HDF5AnnData') %>% downsample_adata(downsample = args$downsample, cell_type = ct)

    # extract all chromosomes contained in the dataset (may contain insertions like GL000220.1)
    chromosomes = adata$var_names %>% str_split_fixed(':', 2) %>% as_tibble() %>% select(V1) %>% distinct() %>% pull()
    genome.df = data.frame(chromosome = chromosomes, stringsAsFactors = FALSE)
    # fetch chromosome lengths for hg38
    genome_info = GenomeInfoDb::Seqinfo(genome = 'hg38')
    genome.df$length = sapply(genome.df$chromosome, function(chrom) {
        if (chrom %in% c('chrM', 'MT')) {
            return(16569) # mitochondrial genome length
        } else if (chrom %in% rownames(genome_info %>% as.data.frame())) {
            return(seqlengths(genome_info)[chrom])
        } else {
            # for alternative contigs, fetch from GenomeInfoDb package, where row names are in the format 'chr1_KI270706v1_random'
            alt_contig_name = rownames(genome_info)[grepl(paste0('_', chrom, 'v'), rownames(genome_info %>% as.data.frame()))]
            if (length(alt_contig_name) == 1) {
                return(seqlengths(genome_info)[alt_contig_name])
            } else { # if no match or multiple matches, return NA
                return(NA)
            }
        }
    }) %>% drop_na()

    # subset to only include peaks where a chromosome length is available
    adata = adata[, adata$var_names %>% str_split_fixed(':', 2) %>% as_tibble() %>% select(V1) %>% pull() %in% genome.df$chromosome]
    X = adata$X
    counts = X %>% t() %>% as('CsparseMatrix') # transpose to have peaks as rows and cells as columns

    # Build Monocle3 CDS
    peak_metadata = adata$var %>% rownames_to_column('peak') %>% mutate(peak = gsub(':|-', '_', peak)) %>% 
        column_to_rownames('peak') %>% as.data.frame()
    rownames(counts) = rownames(peak_metadata)
    cell_metadata = adata$obs %>% as.data.frame()
    cds = new_cell_data_set(
        expression_data = counts,
        cell_metadata = cell_metadata,
        gene_metadata = peak_metadata
    )

    # Search for dimensionality reduction results in the following order: UMAP, tSNE, LSI
    if ('X_umap' %in% names(adata$obsm)) {
        umap = adata$obsm[['X_umap']]
        reducedDims(cds)$UMAP = as.matrix(umap)
        cicero_cds = make_cicero_cds(cds, reduced_coordinates = reducedDims(cds)$UMAP)
    } else if ('X_tsne' %in% names(adata$obsm)) {
        tsne = adata$obsm[['X_tsne']]
        reducedDims(cds)$tSNE = as.matrix(tsne)
        cicero_cds = make_cicero_cds(cds, reduced_coordinates = reducedDims(cds)$tSNE)
    } else if ('LSI' %in% names(adata$obsm)) {
        lsi = adata$obsm[['LSI']]
        reducedDims(cds)$LSI = as.matrix(lsi)
        cicero_cds = make_cicero_cds(cds, reduced_coordinates = reducedDims(cds)$LSI)
    } else {
        stop('No dimensionality reduction results found in the input AnnData object.')
    }

    # run cicero
    if (! file.exists(paste0(args$out, '.cicero.connections.txt.gz')) | args$force) {
        connections = run_cicero(cicero_cds,genomic_coords = genome.df)
        connections %>% write_tsv(paste0(args$out, '.cicero.connections.txt.gz'))
    } else {
        connections = read_tsv(paste0(args$out, '.cicero.connections.txt.gz'), col_types = cols())
    }
    message('Cicero connections saved to: ', paste0(args$out, '.cicero.connections.txt.gz'))

    # identify cis co-accessible networks (CCANs)
    if (! file.exists(paste0(args$out, '.cicero.ccans.txt')) | args$force) {
        CCANs = generate_ccans(connections)
        CCANs %>% write_tsv(paste0(args$out, '.cicero.ccans.txt'))
    } else {
        CCANs = read_tsv(paste0(args$out, '.cicero.ccans.txt'), col_types = cols())
    }
    message('CCANs saved to: ', paste0(args$out, '.cicero.ccans.txt'))
    # CCANs is a dataframe with columns: peak, CCAN; 
    # for downstream analysis, use 
    # CCANs %>% group_by(CCAN) %>% summarise(peaks = list(peak)) to get a list of peaks for each CCAN

}

#### ChromVAR main code ####
run.chromvar = function(args) {
    # not yet implemented
    stop('ChromVAR analysis is not yet implemented in this script.')
}

#### Main function ####
main = function(args) {
    if (args$cicero) {
        run.cicero(args)
    }
    if (args$chromvar) {
        run.chromvar(args)
    }
}
main(args)