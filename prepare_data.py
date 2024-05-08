import scanpy as sc
from pathlib import Path

#
# make adata smaller by selecting top 1500 genes (otherwise my laptop blows) and split into trainval and test by samples (dataset has no batch effect)
# test set consist of 1x Healthy patient, 1x IPF patient, 1x public databse. It is approx. 21% of the entire dataset
#

file_path = Path("/Users/mathias/Code/masterpraktikum_ssl/Preprocess_toyST/adata_vis_human_spatial_paper.h5ad")
adata = sc.read(file_path)
sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=1500, subset=False)
variable_genes = adata.var.highly_variable
adata = adata[:, variable_genes]

samples_trainval =  ['90_A1_H237762_IPF_processed_CM', '90_C1_RO-730_Healthy_processed_CM', '91_B1_RO-728_Healthy_processed_CM', '91_D1_24513-17_IPF_processed_CM', '92_A1_RO-3203_Healthy_processed_CM', '1217_0001_processed_aligned', '1217_0002_processed_aligned', '1217_0003_processed_aligned']
samples_test = ['91_A1_RO-727_Healthy_processed_CM', '92_D1_RO-3736_IPF_processed_CM', '1217_0004_processed_aligned']
adata_trainval = adata[adata.obs['sample'].isin(samples_trainval)]
adata_test = adata[adata.obs['sample'].isin(samples_test)]

adata_trainval.write_h5ad(
    "data/adata_trainval_uncompressed.h5ad"
)
adata_test.write_h5ad(
    "data/adata_test_uncompressed.h5ad"
)
