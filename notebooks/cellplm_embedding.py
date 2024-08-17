from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
from masterpraktikum_ssl_segmentation.DM2C.code.utils import recursive_file_list
import numpy as np
import anndata as ad
import os

h5ad_PATH = '/p/project1/hai_pathology/subgroup_merel/gex_data/anndata'

all_files = recursive_file_list(h5ad_PATH)
pipeline = CellEmbeddingPipeline(pretrain_prefix='20230926_85M',  # Specify the pretrain checkpoint to load
                                         pretrain_directory='ckpt')
for file in all_files:
    print(os.path.splitext(os.path.basename(file))[0])
    data = ad.read_h5ad(file)
    embedding = pipeline.predict(data, device='cpu')  # Specify a gpu or cpu for model inference
    np.save(os.path.join('/p/project1/hai_pathology/embeddings/gex_embed/', os.path.splitext(os.path.basename(file))[0]), embedding)