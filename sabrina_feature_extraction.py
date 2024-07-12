import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
# import cv2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math

# from dataset import SlideDataset
from models.model import get_models
from models.histaugan.model import HistAuGAN
from models.histaugan.augment import augment
# from utils.utils import (bgr_format, get_driver, get_scaling,
                         # save_qupath_annotation, save_tile_preview, threshold)

parser = argparse.ArgumentParser(description="Feature extraction on patches")

parser.add_argument(
    "--patch_path",
    help="Path where patches are stored",
    type=str,
    required=True
)

parser.add_argument("--patch_size", help="Patch size of patches", default=512, type=int)

parser.add_argument(
    "--save_path",
    help="Path to save the extracted features",
    type=str,
)

parser.add_argument(
    "--models",
    help="Select model names for feature extraction",
    nargs="+",
    type=str,
    required=True
)

parser.add_argument("--batch_size", default=256, type=int)

parser.add_argument(
    "--histaugan", 
    help="if set, use HistAuGAN for tile augmentation",
    action='store_true',
)

def main(args):
    """
    Args:
    args: argparse.Namespace, containing the following attributes:
    - patch_path (str): Path where patches are stored.
    - patch_size (int): Patch size
    - save_path (str): Path where to save the extracted features.
    - models (list): List of models to use for feature extraction.
    - batch_size (int): Batch size for processing patches.

    Returns:
    None
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    patch_dirs = [Path(args.patch_path) / dir_name for dir_name in sorted(Path(args.patch_path).glob('*')) if dir_name.is_dir()]

    model_dicts = get_models(args.models)
    output_path = Path(args.save_path) / "h5_files"
    output_path.mkdir(parents=True, exist_ok=True)

    for model in model_dicts:
        model_name = model["name"]
        save_dir = output_path / f"{args.patch_size}px_{model_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        arg_dict = vars(args)
        with open(save_dir / "config.yml", "w") as f:
            for arg_name, arg_value in arg_dict.items():
                f.write(f"{arg_name}: {arg_value}\n")

    start = time.perf_counter()
    extract_features(patch_dirs, model_dicts, device, args)
    end = time.perf_counter()
    elapsed_time = end - start
    print("Time taken: ", elapsed_time, "seconds")

def patches_to_feature(patch_files: list, model_dicts: list[dict], device: torch.device, args: argparse.Namespace, histaugan=None):
    feats = {model_dict["name"]: [] for model_dict in model_dicts}
    feats_aug = {model_dict["name"]: [] for model_dict in model_dicts} if histaugan else None

    with torch.no_grad():
        for model_dict in model_dicts:
            model = model_dict["model"]
            transform = model_dict["transforms"]
            model_name = model_dict["name"]

            dataset = SlideDataset(patch_files, transform)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1, shuffle=False)

            if histaugan:
                z_attr = torch.randn((histaugan.opts.num_domains, histaugan.dim_attribute)).to(device)

            for batch in dataloader:
                batch = batch.to(device)
                if args.histaugan:
                    batch_aug = []
                    for d in range(histaugan.opts.num_domains):
                        domain = torch.eye(histaugan.opts.num_domains)[d].unsqueeze(0).to(device)
                        with torch.cuda.amp.autocast():
                            batch_augmented = augment(batch, histaugan, domain, z_attr[d].unsqueeze(0))
                            batch_aug.append(model(batch_augmented).cpu().detach())
                    feats_aug[model_name].append(torch.stack(batch_aug))

                features = model(batch.float())
                feats[model_name] += features.cpu().numpy().tolist()

    return feats, feats_aug


def extract_features(
    patch_dirs: list[Path], 
    model_dicts: list[dict], 
    device: torch.device, 
    args: argparse.Namespace
):

    """
    Extract features from a patch using a given model.

    Args:
        args (argparse.Namespace): Arguments containing various processing parameters.
        model_dicts (list[dict]): Dictionary containing the model, transforms, and model name.
        device (torch.device): Device to perform computations on (CPU or GPU).
        patch_dirs (list[Path]): A Path object representing the path where the tile preview image will be saved.

    Returns:
        None
    """
    feats = {model_dict["name"]: [] for model_dict in model_dicts}
    feats_aug = {model_dict["name"]: [] for model_dict in model_dicts} if args.histaugan else None

    # initialize HistAuGAN for augmentation
    if args.histaugan:
        print('Initializing HistAuGAN...')
        histaugan_path = './HistAuGAN_TCGA-CRC_7sites.ckpt'
        histaugan = HistAuGAN.load_from_checkpoint(histaugan_path)
        del histaugan.dis1, histaugan.dis2, histaugan.dis_c, histaugan.enc_a
        histaugan.to(device).eval()
    else:
        histaugan = None

    for patch_dir in patch_dirs:
        patch_files = list(patch_dir.glob('*.tif'))
        slide_name = patch_dir.name
        print(f"Processing {slide_name} with {len(patch_files)} patches")

        patch_feats, patch_feats_aug = patches_to_feature(patch_files, model_dicts, device, args, histaugan)

        for key in patch_feats.keys():

            feats[key].extend(patch_feats[key])
            if patch_feats_aug:
                feats_aug[key] = torch.concat(patch_feats_aug[key], dim=1).cpu().numpy()

        # Save features for each patch in hdf5 files
        if len(model_dicts)>0:
            print("now writing h5 files")
            save_hdf5(args, slide_name, patch_feats, patch_feats_aug)

# new SlideDataset class for patches instead of slides
class SlideDataset(Dataset):
    def __init__(self, patch_files: list[Path], transform=None):
        self.patch_files = patch_files
        self.transform = transform

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        patch_path = self.patch_files[idx]
        patch = Image.open(patch_path)

        if self.transform:
            patch = self.transform(patch)

        patch = np.array(patch)

        if patch.ndim == 2:
            patch = np.stack([patch] * 3, axis=-1)

        # Ensure the patch is in the correct shape (H, W, C)
        if patch.shape[-1] != 3:
            patch = patch.transpose((1, 2, 0))

        patch = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Change to channel-first format and normalize

        return patch

# new save_hadf5 function for patches
def save_hdf5(args, slide_name, feats, feats_aug):
    """
    Save extracted features to an HDF5 file.

    Args:
        args (argparse.Namespace): Arguments containing various processing parameters.
        patch_files (list[Path]): List of patch file paths.
        feats (dict): Extracted features.
        feats_aug (dict): Extracted augmented features.

    Returns:
        None
    """
    import h5py

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    hdf5_file = save_path / f"{slide_name}.h5"
    print(hdf5_file)

    with h5py.File(hdf5_file, 'w') as f:
        for model_name, feature_list in feats.items():

            f.create_dataset(model_name, data=np.array(feature_list))

        if feats_aug:
            for model_name, feature_list in feats_aug.items():
                f.create_dataset(f"{model_name}_aug", data=np.array(feature_list))

    print(f"Features saved to {hdf5_file}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    print("finished extracting the features from the patches")
