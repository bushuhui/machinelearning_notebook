"""
Transformer教程工具函数
包含常用的模型组件、可视化函数和辅助工具
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from typing import Optional, Tuple, List


# ============================
# 核心组件
# ============================

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = query.size()
        
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        
        return output, attention_weights


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class FeedForwardNetwork(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForwardNetwork, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer基本块（Post-LN）"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x1 = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x1)
        x2 = self.norm2(x1 + self.dropout(ff_output))
        
        return x2, attention_weights


class PreLNTransformerBlock(nn.Module):
    """Transformer基本块（Pre-LN）"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(PreLNTransformerBlock, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # 层归一化 + 多头自注意力 + 残差连接
        norm_x = self.norm1(x)
        attn_output, attention_weights = self.self_attention(norm_x, norm_x, norm_x, mask)
        x1 = x + self.dropout(attn_output)
        
        # 层归一化 + 前馈网络 + 残差连接
        norm_x1 = self.norm2(x1)
        ff_output = self.feed_forward(norm_x1)
        x2 = x1 + self.dropout(ff_output)
        
        return x2, attention_weights


# ============================
# 掩码生成函数
# ============================

def create_padding_mask(seq: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """
    创建填充掩码
    
    Args:
        seq: [batch_size, seq_len]
        pad_token: 填充token的值
    
    Returns:
        mask: [batch_size, seq_len, seq_len]
    """
    batch_size, seq_len = seq.size()
    
    # 创建填充位置的掩码
    pad_mask = (seq != pad_token).unsqueeze(1)  # [batch_size, 1, seq_len]
    pad_mask = pad_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
    
    return pad_mask


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    创建因果掩码（下三角矩阵）
    
    Args:
        seq_len: 序列长度
    
    Returns:
        mask: [seq_len, seq_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask


def create_combined_mask(seq: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """
    创建组合掩码（填充掩码 + 因果掩码）
    
    Args:
        seq: [batch_size, seq_len]
        pad_token: 填充token的值
    
    Returns:
        mask: [batch_size, seq_len, seq_len]
    """
    batch_size, seq_len = seq.size()
    
    # 填充掩码
    pad_mask = create_padding_mask(seq, pad_token)
    
    # 因果掩码
    causal_mask = create_causal_mask(seq_len).to(seq.device)
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # 组合掩码
    combined_mask = pad_mask & causal_mask
    
    return combined_mask


# ============================
# 可视化函数
# ============================

def visualize_attention_weights(attention_weights: torch.Tensor, tokens: Optional[List[str]] = None,
                               head_idx: int = 0, layer_idx: int = 0, title: str = "注意力权重"):
    """
    可视化注意力权重
    
    Args:
        attention_weights: [batch_size, num_heads, seq_len, seq_len] 或 
                          [[batch_size, num_heads, seq_len, seq_len], ...] (多层)
        tokens: token列表，用于标注坐标轴
        head_idx: 要可视化的头索引
        layer_idx: 要可视化的层索引（如果是多层）
        title: 图表标题
    """
    # 处理多层情况
    if isinstance(attention_weights, list):
        weights = attention_weights[layer_idx]
    else:
        weights = attention_weights
    
    # 取第一个batch和指定头的权重
    weights = weights[0, head_idx].detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    
    if tokens is not None:
        sns.heatmap(weights, annot=True, cmap='Blues', fmt='.3f',
                    xticklabels=tokens, yticklabels=tokens)
    else:
        sns.heatmap(weights, annot=True, cmap='Blues', fmt='.3f')
    
    plt.title(f'{title} - 层{layer_idx+1} 头{head_idx+1}')
    plt.xlabel('被关注位置 (Key)')
    plt.ylabel('查询位置 (Query)')
    plt.tight_layout()
    plt.show()


def visualize_multi_head_attention(attention_weights: torch.Tensor, num_heads_to_show: int = 4,
                                  tokens: Optional[List[str]] = None, layer_idx: int = 0):
    """
    可视化多个注意力头
    
    Args:
        attention_weights: 注意力权重
        num_heads_to_show: 显示的头数
        tokens: token列表
        layer_idx: 层索引
    """
    if isinstance(attention_weights, list):
        weights = attention_weights[layer_idx]
    else:
        weights = attention_weights
    
    weights = weights[0].detach().cpu().numpy()  # [num_heads, seq_len, seq_len]
    num_heads_to_show = min(num_heads_to_show, weights.shape[0])
    
    fig, axes = plt.subplots(2, (num_heads_to_show + 1) // 2, 
                             figsize=(5 * ((num_heads_to_show + 1) // 2), 8))
    
    if num_heads_to_show <= 2:
        axes = axes.reshape(2, -1)
    
    for i in range(num_heads_to_show):
        row = i // ((num_heads_to_show + 1) // 2)
        col = i % ((num_heads_to_show + 1) // 2)
        
        if tokens is not None:
            sns.heatmap(weights[i], annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=tokens, yticklabels=tokens, ax=axes[row, col])
        else:
            sns.heatmap(weights[i], annot=True, fmt='.2f', cmap='Blues', ax=axes[row, col])
        
        axes[row, col].set_title(f'头 {i+1}')
        axes[row, col].set_xlabel('Key位置')
        axes[row, col].set_ylabel('Query位置')
    
    # 隐藏多余的子图
    for i in range(num_heads_to_show, axes.size):
        row = i // ((num_heads_to_show + 1) // 2)
        col = i % ((num_heads_to_show + 1) // 2)
        axes[row, col].set_visible(False)
    
    plt.suptitle(f'多头注意力权重 - 层{layer_idx+1}', fontsize=16)
    plt.tight_layout()
    plt.show()


def visualize_positional_encoding(pos_encoding: SinusoidalPositionalEncoding, 
                                 seq_len: int = 50, d_model: int = 64):
    """
    可视化位置编码
    
    Args:
        pos_encoding: 位置编码实例
        seq_len: 序列长度
        d_model: 模型维度
    """
    pe = pos_encoding.pe[0, :seq_len, :d_model].detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 位置编码热力图
    im1 = axes[0, 0].imshow(pe.T, cmap='RdBu_r', aspect='auto')
    axes[0, 0].set_title('位置编码热力图')
    axes[0, 0].set_xlabel('位置')
    axes[0, 0].set_ylabel('维度')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 选择几个维度展示波形
    positions = np.arange(seq_len)
    selected_dims = [0, 1, 2, 3, 10, 20]
    
    for i, dim in enumerate(selected_dims):
        if dim < d_model:
            axes[0, 1].plot(positions, pe[:, dim], label=f'维度 {dim}', alpha=0.7)
    
    axes[0, 1].set_title('不同维度的位置编码波形')
    axes[0, 1].set_xlabel('位置')
    axes[0, 1].set_ylabel('编码值')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 不同位置的编码分布
    selected_positions = [0, 5, 10, 15, 20]
    
    for pos in selected_positions:
        if pos < seq_len:
            axes[1, 0].plot(pe[pos, :], label=f'位置 {pos}', alpha=0.7)
    
    axes[1, 0].set_title('不同位置的编码向量')
    axes[1, 0].set_xlabel('维度')
    axes[1, 0].set_ylabel('编码值')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 位置编码的频率分析
    div_terms = [10000 ** (2 * i / d_model) for i in range(d_model // 2)]
    frequencies = [1 / term for term in div_terms]
    
    axes[1, 1].semilogy(frequencies[:min(32, len(frequencies))], 'o-')
    axes[1, 1].set_title('位置编码的频率谱')
    axes[1, 1].set_xlabel('维度对 (i)')
    axes[1, 1].set_ylabel('频率 (log scale)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_training_curves(train_losses: List[float], val_losses: Optional[List[float]] = None,
                        train_accuracies: Optional[List[float]] = None,
                        val_accuracies: Optional[List[float]] = None):
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accuracies: 训练准确率列表
        val_accuracies: 验证准确率列表
    """
    fig, axes = plt.subplots(1, 2 if train_accuracies else 1, figsize=(15, 5))
    
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    
    # 损失曲线
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='训练损失')
    if val_losses:
        axes[0].plot(epochs, val_losses, 'r-', label='验证损失')
    axes[0].set_title('训练过程中的损失变化')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    if train_accuracies and len(axes) > 1:
        axes[1].plot(epochs, train_accuracies, 'b-', label='训练准确率')
        if val_accuracies:
            axes[1].plot(epochs, val_accuracies, 'r-', label='验证准确率')
        axes[1].set_title('训练过程中的准确率变化')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('准确率')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================
# 辅助函数
# ============================

def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> str:
    """获取模型大小（MB）"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return f"{size_mb:.2f} MB"


def initialize_weights(model: nn.Module):
    """初始化模型权重"""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.1)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def create_learning_rate_scheduler(optimizer, warmup_steps: int, d_model: int):
    """
    创建Transformer论文中的学习率调度器
    
    Args:
        optimizer: 优化器
        warmup_steps: 预热步数
        d_model: 模型维度
    
    Returns:
        scheduler: 学习率调度器
    """
    def lr_lambda(step):
        if step == 0:
            step = 1
        return min(step ** (-0.5), step * warmup_steps ** (-1.5)) * (d_model ** (-0.5))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def save_model_checkpoint(model: nn.Module, optimizer, epoch: int, loss: float, 
                         filepath: str, **kwargs):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        loss: 当前损失
        filepath: 保存路径
        **kwargs: 其他要保存的信息
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, filepath)


def load_model_checkpoint(model: nn.Module, optimizer, filepath: str):
    """
    加载模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        filepath: 检查点路径
    
    Returns:
        epoch, loss, extra_info: epoch、损失和其他信息
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    # 提取其他信息
    extra_info = {k: v for k, v in checkpoint.items() 
                  if k not in ['epoch', 'model_state_dict', 'optimizer_state_dict', 'loss']}
    
    return epoch, loss, extra_info


# ============================
# 设置和配置
# ============================

def setup_matplotlib_chinese():
    """设置matplotlib支持中文显示"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")


def set_random_seed(seed: int = 42):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)