import os.path
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pickle

from src.TemporalNodeEmbedding import TemporalNodeEmbedding
from src.dataset import EmbeddingDataset

DATA_DIR = '/Users/ashfaq/Documents/traces/filtered_data_files'
BATCH_SIZE = 1000
MAX_NODE_COUNTS = 100_000
D_EMBED = 128
N_EPOCHS = 20

def read_data():
    dfs = [pd.read_parquet(os.path.join(DATA_DIR, f'data_{i}.parquet')) for i in range(3)]
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df[['u', 'i']].to_numpy(dtype=int)


if __name__ == '__main__':
    data = read_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edges_dataset = EmbeddingDataset(data)
    dataloader = DataLoader(edges_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = TemporalNodeEmbedding(max_node_count=MAX_NODE_COUNTS, d_embed=D_EMBED, device=device)
    model.update_embeddings(dataloader, epochs=N_EPOCHS)
    all_embeddings = model.get_all_embeddings()
    pickle.dump(all_embeddings, open('data/all_embeddings.pkl', 'wb'))
