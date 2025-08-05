import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from layers.Titans_Memory import MemoryEncoder


class Model(nn.Module):
    def __init__(self,
                 input_dim,
                 d_model,
                 n_layers,
                 n_heads,
                 d_ff,
                 contextual_mem_size,
                 persistent_mem_size,
                 output_horizon):
        super().__init__()
        self.encoder = MemoryEncoder(input_dim, d_model, n_layers, n_heads, d_ff, contextual_mem_size, persistent_mem_size)
        self.output_proj = nn.Linear(d_model, output_horizon)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.output_proj(encoded[:, -1, :]) # Predict next horizon


class TestTimeMemoryManager:
    def __init__(self, model, lr = 1e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = lr)
        self.loss_fn = nn.MSELoss()

    def add_context(self, x_context):
        for block in self.model.encoder.layers:
            block.attn.update_contextual_memory(x_context)

    def adapt(self, x_new, y_new, steps = 1):
        self.model.train()
        for _ in range(steps):
            pred = self.model(x_new)
            loss = self.loss_fn(pred, y_new)
            loss.backward()
            self.optimizer.step()
        return loss.item()