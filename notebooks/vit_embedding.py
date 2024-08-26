import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTForImageClassification

#file_names = "/p/project1/hai_pathology/subgroup_merel/subgroup_merel/image_data/control/AVL"
file_names = '/p/project1/hai_pathology/subgroup_merel/image_data/'

#img_dataset = [os.path.join(dp, f) for dp, dn, filenames in os.walk(file_names) for f in
#               filenames]  # filenames recursively


processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224', output_hidden_states=True
)
class img_Dataset(Dataset):
    def __init__(self, img_path):
        self.imgs = self._get_image_paths(img_path)

    def _get_image_paths(self, root_dir):
        image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.tif')):  # Add more extensions if needed
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __getitem__(self, index):
        return self.imgs[index]

    def __len__(self):
        return len(self.imgs)



def infer(img_path):
    img = Image.open(img_path).convert("RGB")
    inputs = processor(
        img, return_tensors="pt"
    )  # preprocesses for correct input format
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states
    return hidden_states[-1][0][0]

path = '/p/project1/hai_pathology/subgroup_merel/image_data/'
img_dataset = img_Dataset(path)
img_loader = DataLoader(dataset=img_dataset,
                        # image loader so that not all images are read into memory for embedding calculation
                        batch_size=128,
                        shuffle=True)
img_embed = []
for imgs in tqdm(img_loader):
    for img in imgs:
        save_path = img.replace('subgroup_merel/image_data/', 'embeddings/img_embed/')
        dir = os.path.dirname(save_path)
        if not os.path.exists(dir): os.makedirs(dir)
        img_embed = np.array(infer(img))
        np.save(save_path, img_embed)


#embeds = [infer(image) for image in img_dataset]

