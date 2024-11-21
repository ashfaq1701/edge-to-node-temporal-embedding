import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, edges):
        """
        Dataset to handle edges.
        :param edges: List of tuples [(u1, v1), (u2, v2), ...]
        """
        self.edges = edges

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        return torch.tensor(self.edges[idx], dtype=torch.long)
