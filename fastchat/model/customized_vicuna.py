import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple
from torch import Tensor
import torch.nn.functional as F
from fastchat.model.token_adapter import PosAdapter, PartialEmbedding
from transformers import LlamaForCausalLM


class MyVicuna(LlamaForCausalLM):
    def __init__(self, config, n_token_llama=32000):
        super(MyVicuna, self).__init__(config)

        embed_dim = self.model.embed_tokens.weight.data.shape[1]
        llm_embed_tokens = nn.Embedding(num_embeddings=n_token_llama,
                                             embedding_dim=embed_dim).to('cuda')

        # if config.init_method == "sinusoidal" or config.init_method == "average_token":
        added_vec_init = torch.zeros([2 * config.canvas_size, embed_dim]).to('cuda')
        self.model.embed_tokens = PartialEmbedding(llm_embed_tokens, added_vec_init)
        
        # else:
        #     self.model.embed_tokens = PosAdapter(llm_embed_tokens, n_token_llama=n_token_llama,
        #                                          canvas_size=config.canvas_size,
        #                                          sinusoidal_dim=config.sinusoidal_dim,
        #                                          n_layers=config.projection_n_layer,
        #                                          hidden_dim=config.projection_hidden_dim)


    def update_embed(self):
        new_embeddings = self.model.embed_tokens.update_embed()
        self.model.embed_tokens = new_embeddings
        print('New token embedding projections have been recorded as vectors')

