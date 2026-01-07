#!/usr/bin/env', 'Rscript
#### Information ####
# This script formats the spatial transcriptomics data by Qian et al. 2025 
# to satisfy gsMap requirements
# Author: Yuankai He (yh464@cam.ac.uk)
# Date:   2025-08-14

library(tidyverse)
library(anndataR)
library(Seurat)
library(zip)
library(Matrix)

setwd('/rds/project/rds-Nl99R8pHODQ/multiomics/raw/qian_2025')
groups = c('gw15','gw20','gw22','gw34')
slices = list(
  gw15 = c('UMB1367_P1', 'UMB1367_O1', 'UMB1117_F1a','UMB1117_F1b','UMB1117_F2a',
          'UMB1117_F2b','UMB1117_O1','UMB1117_P1','UMB1117_T1'),
  gw20 = c('FB080_F1', 'FB080_F2a', 'FB080_F2b', 'FB080_O1a', 'FB080_O1b', 'FB080_O1c', 
           'FB080_O1d', 'FB080_P1a', 'FB080_P1b', 'FB080_P2', 'FB080_T1', 'FB121_F1', 
           'FB121_F2', 'FB121_O1', 'FB121_P1', 'FB121_P2', 'FB121_T1'),
  gw22 = c('FB123_F1', 'FB123_F2', 'FB123_F3', 'FB123_O1', 'FB123_O2', 'FB123_P1_2', 'FB123_P1'),
  gw34 = c('UMB5900_BA123', 'UMB5900_BA17', 'UMB5900_BA18', 'UMB5900_BA22', 'UMB5900_BA4', 
           'UMB5900_BA40a', 'UMB5900_BA40b', 'UMB5900_BA9')
)

for (group in groups) {
  adata = read_h5ad(paste0(group,'.h5ad'))
  obs = adata$obs %>% mutate(annotation = paste(H1_annotation, H2_annotation, H3_annotation, sep ='_'))
  obs$x = adata$obsm$spatial[,1]; obs$y = adata$obsm$spatial[,2]
  rownames(obs) = str_split_i(rownames(obs), '-',1)
  slice_count = list()
  for (slice in slices[[group]]) {
    unzip(paste0(slice,'.zip'), exdir = paste0(tempdir(),'/',slice), junkpaths = T, 
          files = 'cell_by_gene.csv')
    slice_count[[slice]] = read_csv(paste0(tempdir(), '/',slice,'/cell_by_gene.csv'), 
      col_types = list(cell = 'c')) %>% select(-starts_with('Blank'))
    file.remove(paste0(tempdir(), '/',slice,'/cell_by_gene.csv'))
  }
  slice_count = slice_count %>% bind_rows() %>% column_to_rownames('cell')
  joint_idx = intersect(rownames(obs), rownames(slice_count))
  obs = obs[joint_idx,]
  slice_count = slice_count[joint_idx,] %>% as.matrix() %>% Matrix(sparse = T)
  adata$layers[['count']] = slice_count
  adata$obs = obs %>% select(-x, -y)
  adata$obsm$spatial = obs %>% select(x,y) %>% as.matrix()
  write_h5ad(adata, paste0(group, '_full.h5ad'))
}