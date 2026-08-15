import torch
import torch.nn as nn
from learn_nn.train_framework import get_dataset, TensorHandler
from learn_nn.train_framework.tensor_handler import TensorHandlerBTC
from learn_nn.models.transformer.base_component import Embeddings, PositionEncoding, MultiHeadAttention, FeedForword, LayerNorm
from learn_nn.train_framework.dataset.get_tatoeba import PAD_IDX


EPS = 1e-6


# Transformer Encoder
# 输入 -> 词嵌入 -> 位置编码 -> [多头注意力 --残差连接+层归一化--> 前馈 --残差连接+层归一化-->] * 6 -> 输出

class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_model = MultiHeadAttention()
        self.atten_norm_model = LayerNorm()
        self.ff_model = FeedForword()
        self.ff_norm_model = LayerNorm()
    def forward(self, x, padding_mask=None):
        # 自注意力机制，q,k,v都是自己
        q = k = v = x
        attention_output = self.attention_model.forward(q,k,v, padding_mask)
        # 归一化 + 残差连接
        attention_norm_output = self.atten_norm_model(attention_output + x)
        # 前馈
        ff_output = self.ff_model(attention_norm_output)
        ff_norm_output = self.ff_norm_model(ff_output + attention_norm_output)
        
        return ff_norm_output


# 输入格式：[B, T, C]
class TransformerEncoder(nn.Module):
    def __init__(self, src_vocab, tgt_vocab):
        super().__init__()
        self.embedd_model = Embeddings(src_vocab)
        self.pos_embedded = PositionEncoding()
        self.encoder_blocks = nn.ModuleList([EncoderBlock() for _ in range(6)])
        pass

    def forward(self, src):

        # 词嵌入
        embedded = self.embedd_model(src)
        pos_embedded = self.pos_embedded.forward(embedded)
        temp_src = pos_embedded
        # 源序列 padding mask: True=允许关注, False=PAD(屏蔽)
        src_mask = src.ne(PAD_IDX)
        src_mask = src_mask[:,None, None, :]  # [B,1,1,T_src]

        # 多个编码块
        for block in self.encoder_blocks:
            temp_src = block.forward(temp_src, src_mask)
        return temp_src, src_mask    


if __name__ == '__main__':
    pairs, test_pairs, src_vocab, tgt_vocab = get_dataset(min_len=0, max_len=5)
    
    encoder = TransformerEncoder(src_vocab, tgt_vocab)
    
    epoch_batches = TensorHandlerBTC.pairs_to_train_batches(pairs, 5)

    for src_tensor, tgt_input_tensor, tgt_output_tensor in epoch_batches: 
        encoder_output, src_mask = encoder(src_tensor)
        print(f"encoder_output shape: {encoder_output.shape}, src_mask shape: {src_mask.shape}")
        break

    print(f"test in transformer input layer")
