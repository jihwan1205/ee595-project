"""
Dataset utilities for the Flow Matching assignment
Supports CelebA-HQ-256, CelebA-128, and CelebA-64 datasets.
"""

import os
import glob
import random
import shutil
import zipfile
import json
import hashlib

# Set the kagglehub cache directory to ./data
os.environ["KAGGLEHUB_CACHE"] = "./data"

from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from src.utils import tensor_to_pil_image

# Note: kagglehub is required for downloading the dataset
# Install with: pip install kagglehub
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    print("Warning: kagglehub not available. Dataset download will not work. Run `pip install kagglehub`.")


def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
    for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def get_cache_path(directory, split):
    """Get cache file path for validated images."""
    # Create a hash of the directory path for cache filename
    cache_dir = os.path.join(os.path.dirname(directory), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    dir_hash = hashlib.md5(directory.encode()).hexdigest()[:8]
    cache_file = os.path.join(cache_dir, f"validated_images_{dir_hash}_{split}.json")
    return cache_file


def get_directory_signature(directory):
    """Get a signature of the directory (file count and total size) to detect changes."""
    if not os.path.exists(directory):
        return None
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_files = []
    for ext in image_extensions:
        all_files.extend(glob.glob(os.path.join(directory, ext)))
        all_files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    
    file_count = len(set(all_files))
    total_size = sum(os.path.getsize(f) for f in set(all_files) if os.path.exists(f))
    
    return {"file_count": file_count, "total_size": total_size}


def load_validated_images_cache(directory, split):
    """Load validated images from cache if available and valid."""
    cache_file = get_cache_path(directory, split)
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        # Check if cache is still valid by comparing directory signature
        current_sig = get_directory_signature(directory)
        cached_sig = cache_data.get("signature")
        
        if current_sig and cached_sig and current_sig == cached_sig:
            print(f"Loading validated images from cache ({len(cache_data['valid_images'])} images)...")
            return cache_data['valid_images'], cache_data.get('corrupted_count', 0)
        else:
            # Cache is invalid, directory has changed
            return None
    except Exception as e:
        # Cache file is corrupted or invalid
        return None


def save_validated_images_cache(directory, split, valid_images, corrupted_count):
    """Save validated images to cache."""
    cache_file = get_cache_path(directory, split)
    signature = get_directory_signature(directory)
    
    cache_data = {
        "directory": directory,
        "split": split,
        "signature": signature,
        "valid_images": valid_images,
        "corrupted_count": corrupted_count
    }
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

class CelebAHQ256Dataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None, val_split=0.1, seed=42):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        all_images = []
        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(root, ext)))
            all_images.extend(glob.glob(os.path.join(root, '**', ext), recursive=True))
        
        # Remove duplicates and sort for reproducibility
        all_images = sorted(list(set(all_images)))
        
        if len(all_images) == 0:
            raise ValueError(f"No images found in {root}. Please check the dataset path.")
        
        # Split into train/val using a local random generator to avoid affecting global state
        rng = random.Random(seed)
        rng.shuffle(all_images)
        split_idx = int(len(all_images) * (1 - val_split))
        
        if split == "train":
            self.image_paths = all_images[:split_idx]
        else:  # val
            self.image_paths = all_images[split_idx:]
        
        print(f"CelebA-HQ-256 {split} set: {len(self.image_paths)} images")

    def __getitem__(self, idx):
        # Try to load the image, with fallback to next valid image if corrupted
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                path = self.image_paths[idx]
                img = Image.open(path).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                return img
            except Exception as e:
                # If image is corrupted, try next one
                if attempt < max_attempts - 1:
                    idx = (idx + 1) % len(self.image_paths)
                    continue
                else:
                    raise RuntimeError(f"Failed to load image after {max_attempts} attempts: {e}")
    
    def __len__(self):
        return len(self.image_paths)


class CelebA128Dataset(torch.utils.data.Dataset):
    """Dataset for CelebA 128x128 images. Uses same logic as CelebAHQ256Dataset."""
    def __init__(self, root, split, transform=None, val_split=0.1, seed=42):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        all_images = []
        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(root, ext)))
            all_images.extend(glob.glob(os.path.join(root, '**', ext), recursive=True))
        
        # Remove duplicates and sort for reproducibility
        all_images = sorted(list(set(all_images)))
        
        if len(all_images) == 0:
            raise ValueError(f"No images found in {root}. Please check the dataset path.")
        
        # Split into train/val using a local random generator to avoid affecting global state
        rng = random.Random(seed)
        rng.shuffle(all_images)
        split_idx = int(len(all_images) * (1 - val_split))
        
        if split == "train":
            self.image_paths = all_images[:split_idx]
        else:  # val
            self.image_paths = all_images[split_idx:]
        
        print(f"CelebA-128 {split} set: {len(self.image_paths)} images")

    def __getitem__(self, idx):
        # Try to load the image, with fallback to next valid image if corrupted
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                path = self.image_paths[idx]
                img = Image.open(path).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                return img
            except Exception as e:
                # If image is corrupted, try next one
                if attempt < max_attempts - 1:
                    idx = (idx + 1) % len(self.image_paths)
                    continue
                else:
                    raise RuntimeError(f"Failed to load image after {max_attempts} attempts: {e}")
    
    def __len__(self):
        return len(self.image_paths)


class CelebA64Dataset(torch.utils.data.Dataset):
    """Dataset for CelebA 64x64 images. Uses predefined splits if available (training/, validation/, testing/)."""
    def __init__(self, root, split, transform=None, val_split=0.1, seed=42):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        
        # Check if dataset has predefined splits in subdirectories
        training_dir = os.path.join(root, "training")
        validation_dir = os.path.join(root, "validation")
        testing_dir = os.path.join(root, "testing")
        
        # If predefined splits exist, use them directly
        if os.path.exists(training_dir) or os.path.exists(validation_dir) or os.path.exists(testing_dir):
            # Use predefined directory structure
            if split == "train":
                split_dir = training_dir
            elif split == "val":
                split_dir = validation_dir
            elif split == "test":
                split_dir = testing_dir
            else:
                raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'")
            
            if not os.path.exists(split_dir):
                raise ValueError(f"Split directory not found: {split_dir}")
            
            # Get all image files from the split directory
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            all_images = []
            for ext in image_extensions:
                all_images.extend(glob.glob(os.path.join(split_dir, ext)))
                all_images.extend(glob.glob(os.path.join(split_dir, '**', ext), recursive=True))
            
            # Remove duplicates and sort for reproducibility
            all_images = sorted(list(set(all_images)))
            
            if len(all_images) == 0:
                raise ValueError(f"No images found in {split_dir}. Please check the dataset path.")
            
            self.image_paths = all_images
        else:
            # Fallback: search all images and split randomly (original behavior)
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            all_images = []
            for ext in image_extensions:
                all_images.extend(glob.glob(os.path.join(root, ext)))
                all_images.extend(glob.glob(os.path.join(root, '**', ext), recursive=True))
            
            # Remove duplicates and sort for reproducibility
            all_images = sorted(list(set(all_images)))
            
            if len(all_images) == 0:
                raise ValueError(f"No images found in {root}. Please check the dataset path.")
            
            # Split into train/val using a local random generator to avoid affecting global state
            rng = random.Random(seed)
            rng.shuffle(all_images)
            split_idx = int(len(all_images) * (1 - val_split))
            
            if split == "train":
                self.image_paths = all_images[:split_idx]
            else:  # val
                self.image_paths = all_images[split_idx:]
        
        print(f"CelebA-64 {split} set: {len(self.image_paths)} images")

    def __getitem__(self, idx):
        # Try to load the image, with fallback to next valid image if corrupted
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                path = self.image_paths[idx]
                img = Image.open(path).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                return img
            except Exception as e:
                # If image is corrupted, try next one
                if attempt < max_attempts - 1:
                    idx = (idx + 1) % len(self.image_paths)
                    continue
                else:
                    raise RuntimeError(f"Failed to load image after {max_attempts} attempts: {e}")
    
    def __len__(self):
        return len(self.image_paths)


class CelebAHQ256DataModule(object):
    def __init__(self, root="./data/datasets/celebahq256", batch_size=32, num_workers=4, transform=None, val_split=0.1, seed=42):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.image_resolution = 256  # CelebA-HQ-256 is 256x256
        self.val_split = val_split
        self.seed = seed

        if not os.path.exists(self.root) or not self._has_images(self.root):
            self._download_dataset(dir_path=self.root)
        self._set_dataset()
    
    def _has_images(self, dir_path):
        """Check if directory contains image files."""
        if not os.path.exists(dir_path):
            return False
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        for ext in image_extensions:
            if glob.glob(os.path.join(dir_path, ext)) or glob.glob(os.path.join(dir_path, '**', ext), recursive=True):
                return True
        return False
    
    def _download_dataset(self, dir_path):
        """Download CelebA-HQ-256 dataset if not found."""
        os.environ["KAGGLEHUB_CACHE"] = "./data"
        
        if not KAGGLEHUB_AVAILABLE:
            raise ValueError(
                f"CelebA-HQ-256 dataset not found at {dir_path} and kagglehub is not available.\n"
                "Please install kagglehub with: pip install kagglehub\n"
                "Or manually download and extract the dataset to the specified path."
            )
        
        # Try common Kaggle datasets for CelebA-HQ-256
        # Common dataset identifiers to try
        kaggle_datasets = [
            "badasstechie/celebahq-resized-256x256",
            "jessicali9530/celeba-dataset",
            "lamsimon/celebahq",
        ]
        
        print(f"CelebA-HQ-256 dataset not found at {dir_path}. Attempting to download...")
        
        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(dir_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        downloaded = False
        for dataset_id in kaggle_datasets:
            try:
                print(f"Trying to download from Kaggle dataset: {dataset_id}...")
                dataset_path = kagglehub.dataset_download(dataset_id)
                
                # Find the directory containing images (might be nested)
                image_dir = self._find_image_directory(dataset_path)
                
                if image_dir:
                    # Copy or move images to target directory
                    if image_dir != dir_path:
                        if os.path.exists(dir_path):
                            # If target exists but has no images, remove it
                            if not self._has_images(dir_path):
                                shutil.rmtree(dir_path)
                        
                        if not os.path.exists(dir_path):
                            # Copy the image directory to target location
                            shutil.copytree(image_dir, dir_path)
                            print(f"✓ Successfully downloaded CelebA-HQ-256 dataset to {dir_path}")
                        else:
                            print(f"✓ Dataset already exists at {dir_path}")
                    else:
                        print(f"✓ Dataset found at {dir_path}")
                    
                    downloaded = True
                    break
            except Exception as e:
                print(f"Failed to download from {dataset_id}: {e}")
                continue
        
        if not downloaded:
            # If automatic download failed, provide helpful error message
            raise ValueError(
                f"CelebA-HQ-256 dataset not found at {dir_path} and automatic download failed.\n"
                "Please manually download the CelebA-HQ-256 dataset and extract it to:\n"
                f"  {dir_path}\n"
                "The dataset should contain image files (.jpg, .jpeg, or .png) in the root directory or subdirectories.\n"
                "You can find the dataset on Kaggle or other sources."
            )
    
    def _find_image_directory(self, root_path):
        """Find the directory containing image files, handling nested structures."""
        if not os.path.exists(root_path):
            return None
        
        # Check if root directory has images
        if self._has_images(root_path):
            return root_path
        
        # Search in subdirectories (limit depth to avoid infinite loops)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        for root, dirs, files in os.walk(root_path):
            # Check current directory
            for ext in image_extensions:
                if glob.glob(os.path.join(root, ext)):
                    return root
        
        return None

    def _set_dataset(self):
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.image_resolution, self.image_resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        self.train_ds = CelebAHQ256Dataset(self.root, "train", self.transform, val_split=self.val_split, seed=self.seed)
        self.val_ds = CelebAHQ256Dataset(self.root, "val", self.transform, val_split=self.val_split, seed=self.seed)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False)


class CelebA128DataModule(object):
    def __init__(self, root="./data/datasets/celeba128", batch_size=32, num_workers=4, transform=None, val_split=0.1, seed=42):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.image_resolution = 128  # CelebA-128 is 128x128
        self.val_split = val_split
        self.seed = seed

        if not os.path.exists(self.root) or not self._has_images(self.root):
            self._download_dataset(dir_path=self.root)
        self._set_dataset()
    
    def _has_images(self, dir_path):
        """Check if directory contains image files."""
        if not os.path.exists(dir_path):
            return False
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        for ext in image_extensions:
            if glob.glob(os.path.join(dir_path, ext)) or glob.glob(os.path.join(dir_path, '**', ext), recursive=True):
                return True
        return False
    
    def _download_dataset(self, dir_path):
        """Download CelebA-128 dataset if not found."""
        os.environ["KAGGLEHUB_CACHE"] = "./data"
        
        if not KAGGLEHUB_AVAILABLE:
            raise ValueError(
                f"CelebA-128 dataset not found at {dir_path} and kagglehub is not available.\n"
                "Please install kagglehub with: pip install kagglehub\n"
                "Or manually download and extract the dataset to the specified path."
            )
        
        # Kaggle dataset identifier for CelebA 128x128
        kaggle_dataset_id = "kiernanmcguigan/celebadatasetcropped128x128"
        
        print(f"CelebA-128 dataset not found at {dir_path}. Attempting to download...")
        
        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(dir_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        try:
            print(f"Downloading from Kaggle dataset: {kaggle_dataset_id}...")
            dataset_path = kagglehub.dataset_download(kaggle_dataset_id)
            
            # Check for zip files in the downloaded directory
            zip_files = glob.glob(os.path.join(dataset_path, "*.zip"))
            
            if zip_files:
                print(f"Found {len(zip_files)} zip file(s). Extracting...")
                # Extract all zip files to the target directory
                os.makedirs(dir_path, exist_ok=True)
                for zip_file in zip_files:
                    print(f"Extracting {os.path.basename(zip_file)}...")
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        # Get total number of files for progress
                        total_files = len(zip_ref.namelist())
                        print(f"  Total files to extract: {total_files}")
                        
                        # Extract with progress indication
                        extracted = 0
                        for member in zip_ref.namelist():
                            zip_ref.extract(member, dir_path)
                            extracted += 1
                            if extracted % 1000 == 0:  # Print progress every 1000 files
                                print(f"  Extracted {extracted}/{total_files} files...")
                        
                        print(f"  ✓ Extracted {extracted} files from {os.path.basename(zip_file)}")
                print(f"✓ Successfully downloaded and extracted CelebA-128 dataset to {dir_path}")
            else:
                # No zip files, try to find image directory
                image_dir = self._find_image_directory(dataset_path)
                
                if image_dir:
                    # Copy or move images to target directory
                    if image_dir != dir_path:
                        if os.path.exists(dir_path):
                            # If target exists but has no images, remove it
                            if not self._has_images(dir_path):
                                shutil.rmtree(dir_path)
                        
                        if not os.path.exists(dir_path):
                            # Copy the image directory to target location
                            shutil.copytree(image_dir, dir_path)
                            print(f"✓ Successfully downloaded CelebA-128 dataset to {dir_path}")
                        else:
                            print(f"✓ Dataset already exists at {dir_path}")
                    else:
                        print(f"✓ Dataset found at {dir_path}")
                else:
                    raise ValueError(f"Could not find images in downloaded dataset at {dataset_path}")
                    
        except Exception as e:
            # If automatic download failed, provide helpful error message
            raise ValueError(
                f"CelebA-128 dataset not found at {dir_path} and automatic download failed: {e}\n"
                "Please manually download the CelebA-128 dataset using:\n"
                f"  kaggle datasets download -d {kaggle_dataset_id} -p {dir_path}\n"
                f"  unzip {dir_path}/*.zip -d {dir_path}\n"
                "Or extract the dataset to the specified path manually."
            )
    
    def _find_image_directory(self, root_path):
        """Find the directory containing image files, handling nested structures."""
        if not os.path.exists(root_path):
            return None
        
        # Check if root directory has images
        if self._has_images(root_path):
            return root_path
        
        # Search in subdirectories
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        for root, dirs, files in os.walk(root_path):
            # Check current directory
            for ext in image_extensions:
                if glob.glob(os.path.join(root, ext)):
                    return root
        
        return None

    def _set_dataset(self):
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.image_resolution, self.image_resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        self.train_ds = CelebA128Dataset(self.root, "train", self.transform, val_split=self.val_split, seed=self.seed)
        self.val_ds = CelebA128Dataset(self.root, "val", self.transform, val_split=self.val_split, seed=self.seed)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False)


class CelebA64DataModule(object):
    def __init__(self, root="./data/datasets/celeba64", batch_size=32, num_workers=4, transform=None, val_split=0.1, seed=42):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.image_resolution = 64  # CelebA-64 is 64x64
        self.val_split = val_split
        self.seed = seed

        if not os.path.exists(self.root) or not self._has_images(self.root):
            self._download_dataset(dir_path=self.root)
        self._set_dataset()
    
    def _has_images(self, dir_path):
        """Check if directory contains image files."""
        if not os.path.exists(dir_path):
            return False
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        for ext in image_extensions:
            if glob.glob(os.path.join(dir_path, ext)) or glob.glob(os.path.join(dir_path, '**', ext), recursive=True):
                return True
        return False
    
    def _download_dataset(self, dir_path):
        """Download CelebA-64 dataset if not found."""
        os.environ["KAGGLEHUB_CACHE"] = "./data"
        
        if not KAGGLEHUB_AVAILABLE:
            raise ValueError(
                f"CelebA-64 dataset not found at {dir_path} and kagglehub is not available.\n"
                "Please install kagglehub with: pip install kagglehub\n"
                "Or manually download and extract the dataset to the specified path."
            )
        
        # Kaggle dataset identifier for CelebA 64x64
        kaggle_dataset_id = "arnrob/celeba-small-images-dataset"
        
        print(f"CelebA-64 dataset not found at {dir_path}. Attempting to download...")
        
        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(dir_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        try:
            print(f"Downloading from Kaggle dataset: {kaggle_dataset_id}...")
            dataset_path = kagglehub.dataset_download(kaggle_dataset_id)
            
            # Check for zip files in the downloaded directory
            zip_files = glob.glob(os.path.join(dataset_path, "*.zip"))
            
            if zip_files:
                print(f"Found {len(zip_files)} zip file(s). Extracting...")
                # Extract all zip files to the target directory
                os.makedirs(dir_path, exist_ok=True)
                for zip_file in zip_files:
                    print(f"Extracting {os.path.basename(zip_file)}...")
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        # Get total number of files for progress
                        total_files = len(zip_ref.namelist())
                        print(f"  Total files to extract: {total_files}")
                        
                        # Extract with progress indication
                        extracted = 0
                        for member in zip_ref.namelist():
                            zip_ref.extract(member, dir_path)
                            extracted += 1
                            if extracted % 1000 == 0:  # Print progress every 1000 files
                                print(f"  Extracted {extracted}/{total_files} files...")
                        
                        print(f"  ✓ Extracted {extracted} files from {os.path.basename(zip_file)}")
                print(f"✓ Successfully downloaded and extracted CelebA-64 dataset to {dir_path}")
            else:
                # No zip files, try to find image directory
                image_dir = self._find_image_directory(dataset_path)
                
                if image_dir:
                    # Copy or move images to target directory
                    if image_dir != dir_path:
                        if os.path.exists(dir_path):
                            # If target exists but has no images, remove it
                            if not self._has_images(dir_path):
                                shutil.rmtree(dir_path)
                        
                        if not os.path.exists(dir_path):
                            # Copy the image directory to target location
                            shutil.copytree(image_dir, dir_path)
                            print(f"✓ Successfully downloaded CelebA-64 dataset to {dir_path}")
                        else:
                            print(f"✓ Dataset already exists at {dir_path}")
                    else:
                        print(f"✓ Dataset found at {dir_path}")
                else:
                    raise ValueError(f"Could not find images in downloaded dataset at {dataset_path}")
                    
        except Exception as e:
            # If automatic download failed, provide helpful error message
            raise ValueError(
                f"CelebA-64 dataset not found at {dir_path} and automatic download failed: {e}\n"
                "Please manually download the CelebA-64 dataset using:\n"
                f"  kaggle datasets download -d {kaggle_dataset_id} -p {dir_path}\n"
                f"  unzip {dir_path}/*.zip -d {dir_path}\n"
                "Or extract the dataset to the specified path manually."
            )
    
    def _find_image_directory(self, root_path):
        """Find the directory containing image files, handling nested structures."""
        if not os.path.exists(root_path):
            return None
        
        # Check if root directory has images
        if self._has_images(root_path):
            return root_path
        
        # Search in subdirectories
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        for root, dirs, files in os.walk(root_path):
            # Check current directory
            for ext in image_extensions:
                if glob.glob(os.path.join(root, ext)):
                    return root
        
        return None

    def _set_dataset(self):
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.image_resolution, self.image_resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        self.train_ds = CelebA64Dataset(self.root, "train", self.transform, val_split=self.val_split, seed=self.seed)
        self.val_ds = CelebA64Dataset(self.root, "val", self.transform, val_split=self.val_split, seed=self.seed)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=False)


def create_data_module(dataset_name, root=None, batch_size=32, num_workers=4, transform=None, **kwargs):
    """
    Factory function to create the appropriate data module based on dataset name.
    
    Args:
        dataset_name: Name of the dataset ('celebahq256', 'celeba128', or 'celeba64')
        root: Root directory of the dataset (optional, uses defaults if not provided)
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        transform: Optional transform pipeline
        **kwargs: Additional arguments passed to the data module
    
    Returns:
        Data module instance
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'celebahq256' or dataset_name == 'celebahq' or dataset_name == 'celebahq-256':
        if root is None:
            root = "./data/datasets/celebahq256"
        return CelebAHQ256DataModule(root=root, batch_size=batch_size, num_workers=num_workers, transform=transform, **kwargs)
    
    elif dataset_name == 'celeba128' or dataset_name == 'celeba-128':
        if root is None:
            root = "./data/datasets/celeba128"
        return CelebA128DataModule(root=root, batch_size=batch_size, num_workers=num_workers, transform=transform, **kwargs)
    
    elif dataset_name == 'celeba64' or dataset_name == 'celeba-64':
        if root is None:
            root = "./data/datasets/celeba64"
        return CelebA64DataModule(root=root, batch_size=batch_size, num_workers=num_workers, transform=transform, **kwargs)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets: 'celebahq256', 'celeba128', 'celeba64'")


if __name__ == "__main__":
    # Test all datasets
    print("Testing CelebA-HQ-256 dataset...")
    try:
        celebahq_module = create_data_module("celebahq256")
        print(f"  # training images: {len(celebahq_module.train_ds)}")
        print(f"  # validation images: {len(celebahq_module.val_ds)}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\nTesting CelebA-128 dataset...")
    try:
        celeba128_module = create_data_module("celeba128")
        print(f"  # training images: {len(celeba128_module.train_ds)}")
        print(f"  # validation images: {len(celeba128_module.val_ds)}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\nTesting CelebA-64 dataset...")
    try:
        celeba64_module = create_data_module("celeba64")
        print(f"  # training images: {len(celeba64_module.train_ds)}")
        print(f"  # validation images: {len(celeba64_module.val_ds)}")
    except Exception as e:
        print(f"  Error: {e}")