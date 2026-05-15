#!/bin/bash
mamba activate /rds/project/rds-Nl99R8pHODQ/toolbox/gsmap
h5ad=$1
prefix=$2
workdir=/rds/project/rds-Nl99R8pHODQ/multiomics/gsmap
mkdir -p "$workdir/$prefix/find_latent_representations"
mkdir -p "$workdir/$prefix/latent_to_gene"
mkdir -p "$workdir/$prefix/generate_ldscore"
if [ ! -f "$workdir/$prefix/find_latent_representations/${prefix}_add_latent.h5ad" ]; then
  gsmap run_find_latent_representations --workdir /rds/project/rds-Nl99R8pHODQ/multiomics/gsmap --sample_name "$prefix" \
    --input_hdf5_path "$h5ad" --annotation annotation --data_layer count
fi
if [ ! -f "$workdir/$prefix/latent_to_gene/${prefix}_gene_marker_score.feather" ]; then
  gsmap run_latent_to_gene --workdir /rds/project/rds-Nl99R8pHODQ/multiomics/gsmap --sample_name "$prefix" \
    --annotation annotation --latent_representation latent_GVAE --num_neighbour 51 --num_neighbour_spatial 201
fi
for chrom in {1..22}; do
  ld_dir="$workdir/$prefix/generate_ldscore"
  # count entries safely (suppress errors if dir missing)
  count=$(ls -1 "$ld_dir" 2>/dev/null | wc -l)
  chunk_index=$((count - 4))
  chunk_dir="$ld_dir/${prefix}_chunk${chunk_index}"
  target_file="$chunk_dir/${prefix}.${chrom}.l2.ldscore.feather"
  if [ "$chunk_index" -le 0 ] || [ ! -f "$target_file" ]; then
    gsmap run_generate_ldscore --workdir /rds/project/rds-Nl99R8pHODQ/multiomics/gsmap --sample_name "$prefix" \
      --chrom "$chrom" --bfile_root /rds/project/rds-Nl99R8pHODQ/toolbox/gsmap/gsMap_resource/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC \
      --keep_snp_root /rds/project/rds-Nl99R8pHODQ/toolbox/gsmap/gsMap_resource/LDSC_resource/hapmap3_snps/hm \
      --gtf_annotation_file /rds/project/rds-Nl99R8pHODQ/toolbox/gsmap/gsMap_resource/genome_annotation/gtf/gencode.v46lift37.basic.annotation.gtf \
      --enhancer_annotation_file /rds/project/rds-Nl99R8pHODQ/toolbox/gsmap/gsMap_resource/genome_annotation/enhancer/by_tissue/ALL/ABC_roadmap_merged.bed \
      --gene_window_size 50000 --snp_multiple_enhancer_strategy max_mkscore --gene_window_enhancer_priority gene_window_first
  fi
done
