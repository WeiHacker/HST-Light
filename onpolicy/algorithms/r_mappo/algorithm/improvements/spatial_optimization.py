import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.init import xavier_normal_
from onpolicy.algorithms.utils.util import check

class TopologyEncoder(nn.Module):
    """拓扑编码器（物理拓扑+交通拓扑）"""
    def __init__(self, num_nodes, hidden_dim=64, beta=0.5, device='cpu'):
        super().__init__()
        self.physical_adj = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.traffic_adj = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        self.beta = beta
        self.gcn_layers = nn.ModuleList([
            nn.Linear(num_nodes, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])
        self.tpdv = dict(dtype=torch.float32, device=device)
        
    def forward(self, traffic_flow):
        """动态更新交通邻接矩阵并执行GCN
        Args:
            traffic_flow: 实时交通流矩阵 [batch, num_nodes, num_nodes]
        Returns:
            node_embeddings: 节点编码 [batch, num_nodes, hidden_dim]
        """
        # 更新交通邻接矩阵
        self.traffic_adj.data = traffic_flow.mean(dim=0)
        
        # 组合邻接矩阵
        adj = torch.sigmoid(self.physical_adj) + self.beta * torch.sigmoid(self.traffic_adj)
        adj = check(adj).to(**self.tpdv)
        
        # 3层GCN
        x = adj
        for layer in self.gcn_layers:
            x = torch.relu(layer(x))
        return x

class TrafficFlowModel(nn.Module):
    """车流传播模型"""
    def __init__(self, num_nodes, lambda_=0.1, v_norm=10.0, device='cpu'):
        super().__init__()
        self.distance_mat = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.lambda_ = lambda_
        self.v_norm = v_norm
        self.tpdv = dict(dtype=torch.float32, device=device)
        
    def forward(self, current_flow):
        """计算车流传播衰减
        Args:
            current_flow: 当前流量 [batch, num_nodes]
        Returns:
            propagated_flow: 传播后的流量 [batch, num_nodes]
        """
        distances = torch.sigmoid(self.distance_mat) * 1000  # 标准化到0-1000米
        gamma = torch.exp(-self.lambda_ * distances / self.v_norm)
        gamma = check(gamma).to(**self.tpdv)
        return torch.matmul(current_flow.unsqueeze(1), gamma).squeeze(1)

class GreenWaveOptimizer(nn.Module):
    """绿波带优化器"""
    def __init__(self, num_nodes, v_opt=15.0, device='cpu'):
        super().__init__()
        self.phase_diff = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        self.v_opt = v_opt
        self.tpdv = dict(dtype=torch.float32, device=device)
        
    def forward(self, distances):
        """优化相位差
        Args:
            distances: 路口间距矩阵 [num_nodes, num_nodes]
        Returns:
            optimized_phases: 优化后的相位差 [num_nodes]
        """
        distances = check(distances).to(**self.tpdv)
        ideal_diff = distances / self.v_opt
        current_diff = torch.sigmoid(self.phase_diff) * 30  # 限制在0-30秒
        loss = torch.abs(current_diff - ideal_diff).mean()
        return current_diff.diag(), loss

class FeatureFusionLayer(nn.Module):
    """三级特征融合层"""
    def __init__(self, d_model, num_nodes, nhead=4, device='cpu'):
        super().__init__()
        # 局部特征提取(3D卷积)
        self.local_conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3,3,3)),  # 时间×空间×特征
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=(3,3,3)),
            nn.ReLU()
        )
        self.local_proj = nn.Linear(16, d_model)
        
        # 区域特征(GCN)
        self.region_gcn = TopologyEncoder(num_nodes, d_model, device=device)
        
        # 全局特征(多头注意力池化)
        self.global_attn = nn.MultiheadAttention(d_model, nhead)
        self.global_proj = nn.Linear(d_model, d_model)
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(3*d_model, 3),
            nn.Softmax(dim=-1)
        )
        self.tpdv = dict(dtype=torch.float32, device=device)

    def forward(self, x, adj):
        """
        Args:
            x: 输入特征 [B, T, N, C]
            adj: 邻接矩阵 [N, N]
        Returns:
            fused_feat: 融合后的特征 [B, d_model]
        """
        B, T, N, C = x.shape
        
        # 局部特征
        local_feat = self.local_conv(x.unsqueeze(1))  # [B,1,T,N,C]->[B,16,T',N',C']
        local_feat = local_feat.mean(dim=(2,3,4))  # 全局平均池化
        local_feat = self.local_proj(local_feat)
        
        # 区域特征 
        region_feat = self.region_gcn(x.permute(1,0,2,3).reshape(B*T, N, C))
        region_feat = region_feat.mean(dim=1).reshape(B, T, -1).mean(dim=1)
        
        # 全局特征
        global_feat = x.reshape(B, T*N, C).permute(1,0,2)  # [T*N,B,C]
        global_feat, _ = self.global_attn(global_feat, global_feat, global_feat)
        global_feat = global_feat.mean(dim=0)  # [B,C]
        global_feat = self.global_proj(global_feat)
        
        # 门控融合
        gates = self.gate(torch.cat([local_feat, region_feat, global_feat], dim=-1))
        fused_feat = gates[:,0:1]*local_feat + gates[:,1:2]*region_feat + gates[:,2:3]*global_feat
        
        return fused_feat

def spatial_block(block):
    """空间优化模块的通用构建块"""
    if isinstance(block, list):
        return nn.Sequential(*block)
    return block
