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
parser$add_argument('-f','--force', action = 'store_true', help = 'Force overwrite')

args = parser$parse_args(commandArgs(TRUE))
args$input = normalizePath(args$input)
args$out = normalizePath(args$out, mustWork = FALSE)
if (args$cicero & args$chromvar) {stop("Please choose either --cicero or --chromvar, not both.")}
print('Input arguments:')
print(args)

#### Cicero main code ####
run.cicero = function(args) { # run_cicero is a cicero package function so use dot
    library(anndataR)
    library(Matrix)
    library(monocle3)
    library(cicero)
    library(tidyverse)

    adata = read_h5ad(args$input)
    X = adata$X
    counts = Matrix::t(X)

    # Build Monocle3 CDS
    peak_metadata = adata$var %>% rownames_to_column("peak") %>% mutate(peak = gsub(":|-", "_", peak)) %>% as_tibble()
    cell_metadata = adata$obs %>% rownames_to_column("cell") %>% as_tibble()
    cds = new_cell_data_set(
        expression_data = counts,
        cell_metadata = cell_metadata,
        gene_metadata = peak_metadata
    )

    # extract all chromosomes contained in the dataset (may contain insertions like GL000220.1)
    chromosomes = peak_metadata %>% select(peak) %>% str_split_fixed("_", 3) %>% as_tibble() %>% select(V1) %>% distinct() %>% pull()
    genome.df = data.frame(chromosome = chromosomes, stringsAsFactors = FALSE)
    # fetch chromosome lengths for hg38
    genome_info = GenomeInfoDb::Seqinfo(genome = "hg38")
    genome.df$length = sapply(genome.df$chromosome, function(chrom) {
        if (chrom %in% c("chrM", "MT")) {
            return(16569) # mitochondrial genome length
        } else if (chrom %in% rownames(genome_info)) {
            return(seqlengths(genome_info)[chrom])
        } else {
            # for alternative contigs, fetch from GenomeInfoDb package, where row names are in the format "chr1_KI270706v1_random"
            alt_contig_name = rownames(genome_info)[grepl(paste0("_", chrom, "v"), rownames(genome_info))]
            if (length(alt_contig_name) == 1) {
                return(seqlengths(genome_info)[alt_contig_name])
            } else { # if no match or multiple matches, return NA
                return(NA)
            }
        }
    })

    # Search for dimensionality reduction results in the following order: UMAP, tSNE, LSI
    if ('X_umap' %in% names(adata$obsm)) {
        umap = adata$obsm[["X_umap"]]
        reducedDims(cds)$UMAP = as.matrix(umap)
        cicero_cds = make_cicero_cds(cds, reduced_coordinates = reducedDims(cds)$UMAP)
    } else if ('X_tsne' %in% names(adata$obsm)) {
        tsne = adata$obsm[["X_tsne"]]
        reducedDims(cds)$tSNE = as.matrix(tsne)
        cicero_cds = make_cicero_cds(cds, reduced_coordinates = reducedDims(cds)$tSNE)
    } else if ('LSI' %in% names(adata$obsm)) {
        lsi = adata$obsm[["LSI"]]
        reducedDims(cds)$LSI = as.matrix(lsi)
        cicero_cds = make_cicero_cds(cds, reduced_coordinates = reducedDims(cds)$LSI)
    } else {
        stop("No dimensionality reduction results found in the input AnnData object.")
    }

    # run cicero
    if (! file.exists(paste0(args$out, "_cicero_connections.txt")) | args$force) {
        connections = run_cicero(cicero_cds,genomic_coords = genome.df)
        connections %>% write_tsv(paste0(args$out, "_cicero_connections.txt"), quote = FALSE)
    } else {
        connections = read_tsv(paste0(args$out, "_cicero_connections.txt"), col_types = cols())
    }
    message("Cicero connections saved to: ", paste0(args$out, "_cicero_connections.txt"))

    # identify cis co-accessible networks (CCANs)
    if (! file.exists(paste0(args$out, "_ccans.txt")) | args$force) {
        CCANs = generate_ccans(connections)
        CCANs %>% write_tsv(paste0(args$out, "_ccans.txt"), quote = FALSE)
    } else {
        CCANs = read_tsv(paste0(args$out, "_ccans.txt"), col_types = cols())
    }
    message("CCANs saved to: ", paste0(args$out, "_ccans.txt"))
    # CCANs is a dataframe with columns: peak, CCAN; 
    # for downstream analysis, use 
    # CCANs %>% group_by(CCAN) %>% summarise(peaks = list(peak)) to get a list of peaks for each CCAN

}

#### ChromVAR main code ####
run.chromvar = function(args) {
    # not yet implemented
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