# GVP implementation from DiffHopp https://github.com/jostorge/diffusion-hopping/tree/main
from typing import Tuple, Union, Optional

import torch
from torch import nn as nn
from torch.nn import functional as F

from src.conv_layer import GVPConvLayer
from src.gvp import GVP, s_V
from src.layer_norm import GVPLayerNorm

import os





# class GVPNetwork(nn.Module):
#     def __init__(
#             self,
#             in_dims: Tuple[int, int],
#             out_dims: Tuple[int, int],
#             hidden_dims: Tuple[int, int],
#             num_layers: int,
#             drop_rate: float = 0.0,
#             attention: bool = False,
#             normalization_factor: float=100.0,
#             aggr: str = "add",
#             activations=(F.silu, None),
#             vector_gate: bool = True,
#             eps=1e-4
#     ) -> None:
#         super().__init__()
#         edge_dims = (1,1)

#         self.eps = eps
#         self.embedding_in = nn.Sequential(
#             GVPLayerNorm(in_dims), 
#             GVP(
#                 in_dims,
#                 hidden_dims,
#                 activations=(None,None),
#                 vector_gate=vector_gate
#             ),
#         )
#         self.embedding_out = nn.Sequential(
#             GVPLayerNorm(hidden_dims),
#             GVP(
#                 hidden_dims,
#                 out_dims,
#                 activations=activations,
#                 vector_gate=vector_gate
#             ),
#         )
#         self.edge_embedding = nn.Sequential(
#             GVPLayerNorm(edge_dims),
#             GVP(
#                 edge_dims,
#                 (hidden_dims[0],1),
#                 activations=(None, None),
#                 vector_gate=vector_gate
#             )
#         )

#         self.layers = nn.ModuleList(
#             [
#                 GVPConvLayer(
#                     hidden_dims,
#                     (hidden_dims[0], 1),
#                     drop_rate=drop_rate,
#                     activations=activations,
#                     vector_gate=vector_gate,
#                     residual=True,
#                     attention=attention,
#                     aggr=aggr,
#                     normalization_factor=normalization_factor,
#                 )
#                 for _ in range(num_layers)
#             ]
#         )

#     def get_edge_attr(self, edge_index, pos) -> s_V:
#         V = pos[edge_index[0]] - pos[edge_index[1]]  # [n_edges, 3]
#         s = torch.linalg.norm(V, dim=-1, keepdim=True)  # [n_edges, 1]
#         V = (V / torch.clip(s, min=self.eps))[..., None, :]  # [n_edges, 1, 3]
#         return s, V
    
#     def forward(self, h, pos, edge_index) -> s_V:
#         edge_attr = self.get_edge_attr(edge_index, pos)
#         edge_attr = self.edge_embedding(edge_attr)

#         h = self.embedding_in(h)
#         for layer in self.layers:
#             h = layer(h, edge_index, edge_attr)
        
#         return self.embedding_out(h)

class GVPNetwork(nn.Module):
    def __init__(
            self,
            in_dims: Tuple[int, int],
            out_dims: Tuple[int, int],
            hidden_dims: Tuple[int, int],
            num_layers: int,
            drop_rate: float = 0.0,
            attention: bool = False,
            normalization_factor: float=100.0,
            aggr: str = "add",
            activations=(F.silu, None),
            vector_gate: bool = True,
            eps=1e-4,
            pretrained_weights: Optional[str] = None
    ) -> None:
        super().__init__()
        edge_dims = (1,1)

        self.eps = eps
        self.embedding_in = nn.Sequential(
            GVPLayerNorm(in_dims), 
            GVP(
                in_dims,
                hidden_dims,
                activations=(None,None),
                vector_gate=vector_gate
            ),
        )
        self.embedding_out = nn.Sequential(
            GVPLayerNorm(hidden_dims),
            GVP(
                hidden_dims,
                out_dims,
                activations=activations,
                vector_gate=vector_gate
            ),
        )
        self.edge_embedding = nn.Sequential(
            GVPLayerNorm(edge_dims),
            GVP(
                edge_dims,
                (hidden_dims[0],1),
                activations=(None, None),
                vector_gate=vector_gate
            )
        )

        self.layers = nn.ModuleList(
            [
                GVPConvLayer(
                    hidden_dims,
                    (hidden_dims[0], 1),
                    drop_rate=drop_rate,
                    activations=activations,
                    vector_gate=vector_gate,
                    residual=True,
                    attention=attention,
                    aggr=aggr,
                    normalization_factor=normalization_factor,
                )
                for _ in range(num_layers)
            ]
        )

        # 添加额外的 GVPConvLayer 层
        if pretrained_weights is not None:
            self.transfer_layer = GVPConvLayer(
                hidden_dims,
                (hidden_dims[0], 1),
                drop_rate=drop_rate,
                activations=activations,
                vector_gate=vector_gate,
                residual=True,
                attention=attention,
                aggr=aggr,
                normalization_factor=normalization_factor,
            )

            # 使用预训练权重初始化新层
            if os.path.exists(pretrained_weights):
                print("Pretrained weights found, loading...")
                state_dict = torch.load(pretrained_weights)
                self.transfer_layer.load_state_dict(state_dict, strict=False)
                print("Pretrained weights loaded!")
            else:
                print(f"Pretrained weights not found at {pretrained_weights}")

            # 冻结参数
            for param in self.transfer_layer.parameters():
                param.requires_grad = False

    def get_edge_attr(self, edge_index, pos) -> s_V:
        V = pos[edge_index[0]] - pos[edge_index[1]]  # [n_edges, 3]
        s = torch.linalg.norm(V, dim=-1, keepdim=True)  # [n_edges, 1]
        V = (V / torch.clip(s, min=self.eps))[..., None, :]  # [n_edges, 1, 3]
        return s, V
    
    def forward(self, h, pos, edge_index) -> s_V:
        edge_attr = self.get_edge_attr(edge_index, pos)
        edge_attr = self.edge_embedding(edge_attr)

        h = self.embedding_in(h)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        
        # 如果使用了预训练权重，则将其传递到新层
        if hasattr(self, 'transfer_layer'):
            h = self.transfer_layer(h, edge_index, edge_attr)

        return self.embedding_out(h)
    
    
    
# src/consistency_model.py

import torch
import torch.nn as nn

class ConsistencyModel(nn.Module):
    def __init__(self, base_model, gamma_network, norm_values):
        super().__init__()
        self.base_model = base_model      # dynamics 或 GVP 结构
        self.gamma = gamma_network        # 同 EDM 使用的 gamma 网络
        self.norm_values = norm_values

    def alpha(self, gamma):
        return torch.sqrt(torch.sigmoid(-gamma))

    def sigma(self, gamma):
        return torch.sqrt(torch.sigmoid(gamma))

    def forward(self, x0, t, ligand_diff, batch_seg, ligand_group=None):
        """模拟 diffusion 模型中添加噪声，然后一步预测 x0"""
        gamma_t = self.gamma(t)
        alpha_t = self.alpha(gamma_t)
        sigma_t = self.sigma(gamma_t)

        noise = torch.randn_like(x0) * ligand_diff
        xt = alpha_t[batch_seg] * x0 + sigma_t[batch_seg] * noise
        xt = xt * ligand_diff + x0 * (1 - ligand_diff)  # 保留 context 部分

        x0_pred = self.base_model(
            xh=xt,
            t=t,
            ligand_diff=ligand_diff,
            ligand_group=ligand_group,
            batch_seg=batch_seg,
        )
        x0_pred = x0_pred * ligand_diff  # 仅预测 ligand 部分
        return x0_pred

