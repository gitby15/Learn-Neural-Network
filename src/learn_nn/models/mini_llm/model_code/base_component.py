import torch
import torch.nn as nn
import torch.nn.functional as F
import math
HEAD_COUNT = 4
HEAD_SIZE = 32
EMBEDDING_SIZE = HEAD_COUNT*HEAD_SIZE
DROPOUT_RATE = 0.1


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
        self.linear = nn.Linear(EMBEDDING_SIZE, vocab_size)
    def forward(self,x):
        return self.linear(x)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.w2 = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.wo = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
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

class MHAAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_count = HEAD_COUNT
        self.head_size = HEAD_SIZE
        
        self.w_q = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.w_k = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.w_v = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.w_o = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.atten = None
    def forward(self, q,k,v, is_causal=False):
        # [B, T, C]
        b = q.size(0)
        t_q = q.size(1)
        t_kv = k.size(1)
        h = self.head_count
        d = self.head_size
        
        query = self.w_q(q).view(b,t_q, h,d).transpose(1,2)
        key = self.w_k(k).view(b,t_kv, h,d).transpose(1,2)
        value = self.w_v(v).view(b,t_kv, h,d).transpose(1,2)
        attention = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attention = attention.transpose(1,2).contiguous().view(b, t_q, h*d)
        output = self.w_o(attention)
        return output


class GQAAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x, mask=None):
        return x
        


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # self.gqa = GQAAttention()
        self.mha = MHAAttention()
        self.ffn = FeedForward()
    def forward(self, x, is_causal=False):
        # step1 = self.gqa(x, is_causal)
        step1 = self.mha(x,x,x, is_causal)
        residual = step1 + x
        step2 = self.ffn(residual)
        return step2 + residual

    @staticmethod
    # 返回[B, T, C]
    def build_tril_mask(seq_len:int):
        tensor =  torch.tril(torch.ones(seq_len, seq_len))
        return tensor.unsqueeze(0).unsqueeze(0)



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
