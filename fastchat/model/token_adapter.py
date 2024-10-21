import math
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps: int, dim: int, rescale_steps: int = 3142):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x: Tensor):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(100) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def coord_init_tensor(num_steps, dim, rescale_steps=3142):
    coord_embed = SinusoidalPosEmb(num_steps=num_steps, dim=dim, rescale_steps=rescale_steps)
    input_ids = torch.tensor(list(range(num_steps)))
    return coord_embed(input_ids)



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=0, hidden_dim=1024):
        super(MLP, self).__init__()
        # self.input_size = input_size
        self.n_hidden_layers = n_layers

        assert n_layers > 0

        if n_layers == 1:
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            layers = [nn.Linear(input_dim, hidden_dim)]
            for i in range(n_layers-1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.layers = nn.ModuleList(layers)

    def forward(self, x):
        if self.n_hidden_layers > 1:
            for l in self.layers[:-1]:
                x = F.softplus(l(x))
            out = self.layers[-1](x)
        else:
            out = self.layers[0](x)
        return out

class PosAdapter(nn.Module):
    def __init__(self, llm_embed, n_token_llama, canvas_size, sinusoidal_dim, n_layers=1, hidden_dim=128):
        super(PosAdapter, self).__init__()
        self.canvas_size = canvas_size
        self.llm_embed = llm_embed
        self.llm_embed_tokens = llm_embed.weight.data
        self.n_token_llama = n_token_llama
        self.token_dim = self.llm_embed_tokens.shape[1]
        self.sinusoidal_dim = sinusoidal_dim

        self.sinusoidal_embed = SinusoidalPosEmb(num_steps=canvas_size, dim=sinusoidal_dim, rescale_steps=canvas_size)
        self.adapter_x = MLP(input_dim=self.sinusoidal_dim, output_dim=self.token_dim, n_layers=n_layers, hidden_dim=hidden_dim)
        self.adapter_y = MLP(input_dim=self.sinusoidal_dim, output_dim=self.token_dim, n_layers=n_layers, hidden_dim=hidden_dim)
        self.adapter_w = MLP(input_dim=self.sinusoidal_dim, output_dim=self.token_dim, n_layers=n_layers, hidden_dim=hidden_dim)
        self.adapter_h = MLP(input_dim=self.sinusoidal_dim, output_dim=self.token_dim, n_layers=n_layers, hidden_dim=hidden_dim)

    def forward(self, input_ids: Tensor):

        diff_idx = input_ids - self.n_token_llama
        # added_mask = (diff_idx >= 0)
        x_mask = torch.floor(diff_idx / self.canvas_size) == 0
        y_mask = torch.floor(diff_idx / self.canvas_size) == 1
        w_mask = torch.floor(diff_idx / self.canvas_size) == 2
        z_mask = torch.floor(diff_idx / self.canvas_size) == 3

        added_idx = torch.where(diff_idx < 0, 0, diff_idx)
        llm_idx = torch.where(diff_idx < 0, input_ids, 0)
        sinusoidal_vec = self.sinusoidal_embed(added_idx)
        pos_x_vec = self.adapter_x(sinusoidal_vec)
        pos_y_vec = self.adapter_y(sinusoidal_vec)
        pos_w_vec = self.adapter_w(sinusoidal_vec)
        pos_h_vec = self.adapter_h(sinusoidal_vec)

        llm_vec = self.llm_embed(llm_idx)

        llm_vec[x_mask, :] = pos_x_vec[x_mask, :]
        llm_vec[y_mask, :] = pos_y_vec[y_mask, :]
        llm_vec[w_mask, :] = pos_w_vec[w_mask, :]
        llm_vec[z_mask, :] = pos_h_vec[z_mask, :]

        return llm_vec

    def get_new_embed(self):
        canvas_ids = torch.tensor(list(range(self.canvas_size)))
        sinusoidal_vec = self.sinusoidal_embed(canvas_ids)
        pos_x_vec = self.adapter_x(sinusoidal_vec)
        pos_y_vec = self.adapter_y(sinusoidal_vec)
        pos_w_vec = self.adapter_w(sinusoidal_vec)
        pos_h_vec = self.adapter_h(sinusoidal_vec)

        return torch.cat([pos_x_vec, pos_y_vec, pos_w_vec, pos_h_vec], dim=0)

    def update_embed(self):
        new_embeddings = nn.Embedding(self.n_token_llama + 4 * self.canvas_size, self.token_dim)
        new_embeddings.to(self.llm_embed.weight.device, dtype=self.llm_embed.weight.dtype)
        new_embeddings.weight.data[:self.n_token_llama, :] = self.llm_embed.weight.data[:self.n_token_llama, :]
        new_embeddings.weight.data[self.n_token_llama:, :] = self.get_new_embed()
        return new_embeddings


class PartialEmbedding(nn.Module):
    def __init__(self, llm_embed, added_vec):
        super(PartialEmbedding, self).__init__()
        self.token_dim = llm_embed.weight.data.shape[1]
        self.llm_embed = llm_embed
        self.embed_vec_freeze = llm_embed.weight.data.to('cuda')
        self.weights_train = nn.Parameter(added_vec.to('cuda'))
        self.embed_vec = torch.cat((self.embed_vec_freeze, self.weights_train), 0)
        self.n_vocab = self.embed_vec.shape[0]

    def forward(self, idx):
        lookup = F.embedding(idx, self.embed_vec)
        return lookup

    def update_embed(self):
        new_embeddings = nn.Embedding(self.n_vocab, self.token_dim)
        new_embeddings.to(self.llm_embed.weight.device, dtype=self.llm_embed.weight.dtype)
        new_embeddings.weight.data = self.embed_vec.clone()
        return new_embeddings


if __name__ == "__main__":

    coord_embed = SinusoidalPosEmb(num_steps=128, dim=32)
    input_ids = torch.tensor(list(range(128)))
    print(coord_embed(input_ids).shape)

    llm_vocab_size = 100
    llm_embed = nn.Embedding(num_embeddings=llm_vocab_size + 20, embedding_dim=256)
    m = PosAdapter(llm_embed, n_token_llama=llm_vocab_size, canvas_size=16, sinusoidal_dim=32)
    print(m)

    input_ids = torch.tensor([1, 2, 100, 101, 45, 134])
    print(m(input_ids))


    print(131008000/32000)



