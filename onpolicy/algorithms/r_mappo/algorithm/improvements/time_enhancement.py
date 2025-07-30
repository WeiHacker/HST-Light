import torch
import torch.nn as nn
import numpy as np
import math
from math import sqrt
from onpolicy.algorithms.utils.util import check

class DynamicWindowLSTM(nn.Module):
    """动态窗口分割LSTM模块"""
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, device='cpu'):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.base_window = 5  # 基础窗口大小
        self.alpha = 0.1  # 窗口调整系数
        self.tpdv = dict(dtype=torch.float32, device=device)
        
    def forward(self, flow_history):
        """预测窗口大小变化量
        Args:
            flow_history: 历史车流量 [batch, seq_len, 1]
        Returns:
            window_size: 动态窗口大小 [batch]
        """
        flow_history = check(flow_history).to(**self.tpdv)
        _, (h_n, _) = self.lstm(flow_history)
        delta_flow = self.fc(h_n[-1])  # 预测流量变化
        window_size = self.base_window + self.alpha * delta_flow
        return torch.clamp(window_size, min=3, max=10).int()

class EnhancedPositionalEncoding(nn.Module):
    """增强位置编码：基础+周期+事件编码"""
    def __init__(self, d_model=112, max_len=5000, device='cpu'):
        super().__init__()
        self.base_pe = PositionalEncoding(d_model, max_len, device)  # 基础编码
        self.freq_pe = nn.Linear(1, d_model)  # 周期性编码
        self.event_embed = nn.Embedding(2, d_model)  # 事件编码
        
    def forward(self, x, period, events):
        """
        Args:
            x: 输入序列 [seq_len, batch, d_model]
            period: 周期相位 [batch]
            events: 事件标记 [seq_len, batch]
        """
        base = self.base_pe(x)
        freq = self.freq_pe(period.unsqueeze(-1)).unsqueeze(0)
        event = self.event_embed(events)
        return base + freq + event

class SparseAttentionMask:
    """稀疏注意力掩码生成器"""
    def __init__(self, seq_len, device='cpu'):
        self.seq_len = seq_len
        self.tpdv = dict(dtype=torch.float32, device=device)
        
    def local_window_mask(self, window_size):
        """局部窗口掩码"""
        mask = torch.ones(self.seq_len, self.seq_len)
        for i in range(self.seq_len):
            start = max(0, i-window_size//2)
            end = min(self.seq_len, i+window_size//2+1)
            mask[i, start:end] = 0
        return mask.to(**self.tpdv)
    
    def periodic_mask(self, phase, period):
        """周期关联掩码"""
        mask = torch.ones(self.seq_len, self.seq_len)
        for i in range(self.seq_len):
            same_phase = (torch.abs(phase - phase[i]) % period) < 2
            mask[i, same_phase] = 0
        return mask.to(**self.tpdv)
    
    def event_mask(self, events):
        """事件响应掩码"""
        event_idx = torch.where(events == 1)[0]
        mask = torch.ones(self.seq_len, self.seq_len)
        for i in event_idx:
            mask[i, event_idx] = 0
        return mask.to(**self.tpdv)

class PositionalEncoding(nn.Module):
    """基础位置编码(原sumo_nn中的实现)"""
    def __init__(self, d_model=112, max_len=5000, device='cpu'):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.tpdv = dict(dtype=torch.float32, device=device)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] or [seq_len*batch_size, 1, embedding_dim]
        """
        if x.dim() == 3:
            if x.size(1) == 1:  # [seq_len*batch, 1, d_model]
                seq_len = x.size(0)
                pe = self.pe[:seq_len].unsqueeze(1)  # [seq_len, 1, d_model]
            else:  # [seq_len, batch, d_model]
                seq_len = x.size(0)
                pe = self.pe[:seq_len].unsqueeze(1)  # [seq_len, 1, d_model]
                pe = pe.expand(-1, x.size(1), -1)  # [seq_len, batch, d_model]
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        x = x + pe.to(x.device)
        return self.dropout(x)
