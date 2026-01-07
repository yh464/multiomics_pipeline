#!/bin/bash
# mamba activate /rds/project/rds-Nl99R8pHODQ/toolbox/gsmap
dataset=$1
dataset=${dataset/.h5ad/}
workdir=/rds/project/rds-Nl99R8pHODQ/multiomics/gsmap
if [ ! -f $workdir/$dataset/find_latent_representations/${dataset}_add_latent.h5ad ]; then
  gsmap run_find_latent_representations --workdir /rds/project/rds-Nl99R8pHODQ/multiomics/gsmap --sample_name $dataset \
    --input_hdf5_path /rds/project/rds-Nl99R8pHODQ/multiomics/spatial/${dataset/_//}.h5ad --annotation annotation --data_layer count
fi
if [ ! -f $workdir/$dataset/latent_to_gene/${dataset}_gene_marker_score.feather ]; then
  gsmap run_latent_to_gene --workdir /rds/project/rds-Nl99R8pHODQ/multiomics/gsmap --sample_name $dataset \
    --annotation annotation --latent_representation latent_GVAE --num_neighbour 51 --num_neighbour_spatial 201
fi
for chrom in {1..22}; do
  if [ ! -f $workdir/$dataset/generate_ldscore/${dataset}_chunk$(($(ls $workdir/$dataset/generate_ldscore | wc -l)-4))/$dataset.$chrom.l2.ldscore.feather ]; then
    gsmap run_generate_ldscore --workdir /rds/project/rds-Nl99R8pHODQ/multiomics/gsmap --sample_name $dataset \
      --chrom $chrom --bfile_root /rds/project/rds-Nl99R8pHODQ/toolbox/gsmap/gsMap_resource/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC \
      --keep_snp_root /rds/project/rds-Nl99R8pHODQ/toolbox/gsmap/gsMap_resource/LDSC_resource/hapmap3_snps/hm \
      --gtf_annotation_file /rds/project/rds-Nl99R8pHODQ/toolbox/gsmap/gsMap_resource/genome_annotation/gtf/gencode.v46lift37.basic.annotation.gtf \
      --enhancer_annotation_file /rds/project/rds-Nl99R8pHODQ/toolbox/gsmap/gsMap_resource/genome_annotation/enhancer/by_tissue/ALL/ABC_roadmap_merged.bed \
      --gene_window_size 50000 --snp_multiple_enhancer_strategy max_mkscore --gene_window_enhancer_priority gene_window_first
  fi
done
