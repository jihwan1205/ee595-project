import torch
import numpy as np
import os


class ReflowDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.pair_paths = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith('.pt')])

    def __len__(self):
        return len(self.pair_paths)

    def __getitem__(self, idx):
        sample_path = self.pair_paths[idx]
        sample = torch.load(sample_path)
        return sample