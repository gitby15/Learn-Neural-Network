# 因为我太穷了，买不起卡，所以我不得不先提前学习几个模型训练性能优化的方案
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


QUERY_HEAD_COUNT = 8
KV_HEAD_COUNT = int(QUERY_HEAD_COUNT / 4)
HEAD_SIZE = 64

EMBEDDING_SIZE = QUERY_HEAD_COUNT * HEAD_SIZE

# 推荐是2.7倍，但是不知道为啥
FFN_HIDDEN_SIZE = int(EMBEDDING_SIZE * 8 / 3)
FFN_HIDDEN_SIZE = ((FFN_HIDDEN_SIZE + 63) // 64) * 64

DROPOUT_RATE = 0.1
ATTENTION_LAYER_COUNT = 6


# 词嵌入，输入的词表需要是BPE后的词表
# 旋转位置编码(timformer用的是绝对位置编码，优缺点可以上B站搜一下)
class Embedding(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE)
    def forward(self, x):
        return self.embedding(x)

    def _test_fill_weights(self, value: float):
        self.embedding.weight.data.fill_(value)


class Ropa(nn.Module):
    def __init__(self, dim=EMBEDDING_SIZE, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x):
        return self.ropa_embedding(x)

    def ropa_embedding(self, x):
        seq_len = x.size(-2)
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        x_rotated = torch.cat((-x[..., self.dim // 2:], x[..., :self.dim // 2]), dim=-1)
        return x * cos + x_rotated * sin

class PosEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return self.cross_product_embedding(x)

    def cross_product_embedding(self, x):
        # 输入[B, T, C]
        # 输出[B, T, C]
        seq_len = x.size(1)
        pos_encoding = torch.zeros(seq_len, EMBEDDING_SIZE)
        pos_idx = torch.arange(0, seq_len).unsqueeze(1)

        div_deno = torch.exp(torch.arange(0, EMBEDDING_SIZE, 2) * (-math.log(10000.0) / EMBEDDING_SIZE))
        position_value = pos_idx * div_deno
        pos_encoding[:, 0::2] = torch.sin(position_value)
        pos_encoding[:, 1::2] = torch.cos(position_value)
        # 绝对位置编码，是直接加上编码值
        output = x + pos_encoding
        return output
    
    # Todo: 实现旋转位置编码
    def ropa_embedding(self,x):
        pass


class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.rms_norm = nn.RMSNorm(EMBEDDING_SIZE)
    def forward(self, x):
        return self.rms_norm(x)

class Linear(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.linear = nn.Linear(EMBEDDING_SIZE, vocab_size, bias=False)
    def forward(self,x):
        return self.linear(x)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(EMBEDDING_SIZE, FFN_HIDDEN_SIZE, bias=False)
        self.w2 = nn.Linear(EMBEDDING_SIZE, FFN_HIDDEN_SIZE, bias=False)
        self.wo = nn.Linear(FFN_HIDDEN_SIZE, EMBEDDING_SIZE, bias=False)
        self.dropout = nn.Dropout(DROPOUT_RATE)
    def forward(self, x):
        output1 = self.w1(x)
        output1 = F.silu(output1)
        output2 = self.w2(x)
        wo_input = output1 * output2
        wo_output = self.wo(wo_input)
        wo_output = self.dropout(wo_output)
        return wo_output




def scaled_dot_product_attention(q,k,v,mask=None,is_causal=False):
    # Todo: 自己实现一遍，深入理解
    pass

class GQAAttention(nn.Module):
    def __init__(self, rope: Ropa):
        super().__init__()
        self.query_head_count = QUERY_HEAD_COUNT
        self.kv_head_count = KV_HEAD_COUNT
        self.head_size = HEAD_SIZE
        
        self.w_q = nn.Linear(EMBEDDING_SIZE , self.query_head_count * self.head_size, bias=False)
        self.w_k = nn.Linear(EMBEDDING_SIZE , self.kv_head_count * self.head_size, bias=False)
        self.w_v = nn.Linear(EMBEDDING_SIZE , self.kv_head_count * self.head_size, bias=False)
        self.w_o = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE, bias=False)
        self.atten = None
        self.rope = rope
    def forward(self, q,k,v, is_causal=True):
        # [B, T, C]
        b = q.size(0)
        t_q = q.size(1)
        t_kv = k.size(1)
        qh = self.query_head_count
        kv_h = self.kv_head_count
        d = self.head_size
        
        query = self.w_q(q).view(b,t_q, qh,d).transpose(1,2)
        key = self.w_k(k).view(b,t_kv, kv_h,d).transpose(1,2)
        value = self.w_v(v).view(b,t_kv, kv_h,d).transpose(1,2)
        query = self.rope(query)
        key = self.rope(key)
        attention = F.scaled_dot_product_attention(query, key, value, is_causal=is_causal, enable_gqa=True)
        attention = attention.transpose(1,2).reshape(b, t_q, qh*d)
        output = self.w_o(attention)
        return output
        


class TransformerBlock(nn.Module):
    def __init__(self, rope: Ropa):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.gqa = GQAAttention(rope)
        self.ffn_norm = RMSNorm()
        self.ffn = FeedForward()
    def forward(self, x, is_causal=True):
        normed = self.attn_norm(x)
        x = x + self.gqa(normed, normed, normed, is_causal)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class AttentionLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        rope = Ropa(dim=HEAD_SIZE)
        self.attention_blocks = nn.ModuleList([
            TransformerBlock(rope) for _ in range(ATTENTION_LAYER_COUNT)
        ])        
    def forward(self, x):
        attention_output = x
        for block in self.attention_blocks:
            attention_output = block(attention_output)
        return attention_output

if __name__ == '__main__':
    
    embedding = Embedding()
    embedding._test_fill_weights(0)
    
    # 输入[B, T], 输出[B,T,C]
    embedding_input = torch.ones(3,5,dtype=torch.long)
    embedding_output = embedding(embedding_input)
    # print(f"embedding_output size: {embedding_output.shape}")

    pos_embedding = PosEmbedding()
    pos_embedding_output = pos_embedding(embedding_output)
    # print(f"pos_embedding_output size: {pos_embedding_output.shape}")
    
    transformer_blocks = nn.ModuleList([
        TransformerBlock() for _ in range(1)
    ])
    transformer_input = pos_embedding_output
    
    for block in transformer_blocks:
        transformer_input = block(transformer_input, True)
    # print(transformer_input)
