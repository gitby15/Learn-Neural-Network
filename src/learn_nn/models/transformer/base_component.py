import torch
import torch.nn as nn
import torch.nn.functional as F
import math

 
HEAD_SIZE = 64
HEAD_COUNT = 8
ATTENTION_SIZE = HEAD_COUNT* HEAD_SIZE


EMBED_SIZE = 512
DROPOUT_RATE = 0.01

MAX_IMPUT_LEN = 512 # 暂定一句话最长只有512个token，诶，有点像LLM的最大上下文长度


class Embeddings(nn.Module):
    def __init__(self, src_vocab: list):
        super().__init__()
        self.src_vocab = src_vocab
        self.vocab_size = len(src_vocab)
        self.embed = nn.Embedding(self.vocab_size, EMBED_SIZE)
        self._embed_scale_factor = math.sqrt(EMBED_SIZE/2)
        self.dropout = nn.Dropout(DROPOUT_RATE)
    def forward(self, x):
        output = self.embed(x)
        # 将输出乘以一个适当的值，避免后面做位置编码的时候，位置编码的信息浓度太高。
        # 位置编码的值在[0-1]之间
        output = output * self._embed_scale_factor
        output = self.dropout(output)
        return output

# 这一层没有神经网络，只是单纯地给每个位置添加一个位置编码向量
class PositionEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        
        pe = torch.zeros(MAX_IMPUT_LEN, EMBED_SIZE)
        position = torch.arange(0, MAX_IMPUT_LEN).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, EMBED_SIZE, 2) * (-math.log(10000.0) / EMBED_SIZE))
        position_value = position * div_term
        pe[:, 0::2] = torch.sin(position_value)
        pe[:, 1::2] = torch.cos(position_value)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(DROPOUT_RATE)
    
    # embedd的向量加上位置编码
    def forward(self, x):
        # x 的形状是 [T, B, C]，所以取 x.size(0) 作为序列长度
        # self.pe[:, :T] 是 [1, T, C]，转置为 [T, 1, C] 以正确广播到 [T, B, C]
        x = x + self.pe[:, :x.size(0)].transpose(0, 1)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        _embed_dim = HEAD_COUNT * HEAD_SIZE
        self.head_count = HEAD_COUNT
        self.head_size = HEAD_SIZE
        self.w_q = nn.Linear(_embed_dim, _embed_dim)
        self.w_k = nn.Linear(_embed_dim, _embed_dim)
        self.w_v = nn.Linear(_embed_dim, _embed_dim)
        self.w_o = nn.Linear(_embed_dim, _embed_dim)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.atten = None

    # Q,K,V的形状是[T, B, C]
    def forward(self, query, key, value, mask = None):
        t_q = query.size(0)
        t_k = key.size(0)
        b = query.size(1)
        h = self.head_count
        d = self.head_size

        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)

        # [T, B, H*D] → [T, B, H, D] → [B, H, T, D]
        query = query.view(t_q, b, h, d).permute(1, 2, 0, 3)
        key = key.view(t_k, b, h, d).permute(1, 2, 0, 3)
        value = value.view(t_k, b, h, d).permute(1, 2, 0, 3)

        # 计算注意力权重: 输出形状 [B, H, T_q, D]
        attention = F.scaled_dot_product_attention(query, key, value, mask)

        # [B, H, T_q, D] → [B, T_q, H, D] → contiguous → [B, T_q, H*D] → [T_q, B, C]
        attention = attention.transpose(1, 2).contiguous().view(b, t_q, h * d)
        attention = attention.transpose(0, 1)
        output = self.w_o(attention)
        output = self.dropout(output)
        return output

class FeedForword(nn.Module):
    def __init__(self):
        super().__init__()
        # 论文上写的是4，这里用2看看效果
        _middle_size = EMBED_SIZE * 2
        self.input_layer = nn.Linear(EMBED_SIZE, _middle_size)
        self.out_put_layer = nn.Linear(_middle_size, EMBED_SIZE)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.out_put_layer(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(EMBED_SIZE, eps=1e-6)

    def forward(self, x):
        output = self.norm(x)
        return output


class FinalOutput(nn.Module):
    def __init__(self, src_vocab, tgt_vocab):
        super().__init__()
        self.linear = nn.Linear(EMBED_SIZE, len(tgt_vocab))
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        output = self.linear(x)
        output = self.dropout(output)
        return output