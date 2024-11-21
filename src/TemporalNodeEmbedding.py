import torch
import torch.nn as nn
import torch.optim as optim


class TemporalNodeEmbedding(object):
    def __init__(
            self,
            max_node_count,
            d_embed,
            device=None,
            margin=1.0
    ):
        self.max_node_count = max_node_count
        self.d_embed = d_embed
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.margin = margin

        self.embedding = nn.Embedding(max_node_count, d_embed)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        self.embedding = self.embedding.to(self.device)

        self.node_id_to_idx = {}
        self.current_idx = 0

        self.optimizer = optim.AdamW(
            self.embedding.parameters(),
            lr=0.0001
        )

    def _get_or_create_idx_batch(self, nodes):
        indices = []
        for node in nodes:
            node_id = node.item() if isinstance(node, torch.Tensor) else int(node)
            if node_id not in self.node_id_to_idx:
                if self.current_idx >= self.max_node_count:
                    raise ValueError("Maximum node count exceeded.")
                self.node_id_to_idx[node_id] = self.current_idx
                self.current_idx += 1
            indices.append(self.node_id_to_idx[node_id])
        return torch.tensor(indices, device=self.device)

    def _compute_loss_batch(self, edges_batch):
        u_nodes = edges_batch[:, 0]
        v_nodes = edges_batch[:, 1]

        u_indices = self._get_or_create_idx_batch(u_nodes.cpu().numpy())
        v_indices = self._get_or_create_idx_batch(v_nodes.cpu().numpy())

        u_embeds = self.embedding(u_indices)
        v_embeds = self.embedding(v_indices)

        loss = torch.norm(u_embeds - v_embeds, p=2, dim=1).mean()

        return loss

    def update_embeddings(self, dataloader, epochs=1):
        self.embedding.train()

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, edges_batch in enumerate(dataloader):
                edges_batch = edges_batch.to(self.device)

                self.optimizer.zero_grad()
                loss = self._compute_loss_batch(edges_batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.embedding.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def get_embedding(self, node_id):
        if node_id not in self.node_id_to_idx:
            raise ValueError("Node ID not found.")
        idx = self.node_id_to_idx[node_id]
        return self.embedding(torch.tensor(idx, device=self.device)).detach().cpu().numpy()

    def get_all_embeddings(self):
        all_indices = torch.arange(0, self.current_idx, device=self.device)
        all_embeddings = self.embedding(all_indices)
        return {
            node_id: all_embeddings[idx].detach().cpu().numpy()
            for node_id, idx in self.node_id_to_idx.items()
        }
