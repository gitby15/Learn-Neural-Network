import torch
import torch.nn as nn
from learn_nn.train_framework import get_dataset, TensorHandler
from learn_nn.models.transformer.base_component import Embeddings, PositionEncoding, MultiHeadAttention, FeedForword, LayerNorm

DROPOUT_RATE = 0.01


# Transformer Encoder
# 输入 -> 词嵌入 -> 位置编码 -> [多头注意力 --残差连接+层归一化--> 前馈 --残差连接+层归一化-->] * 6 -> 输出

class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_model = MultiHeadAttention()
        self.atten_norm_model = LayerNorm()
        self.ff_model = FeedForword()
        self.ff_norm_model = LayerNorm()
        self.dropout = nn.Dropout(DROPOUT_RATE)
    def forward(self, x):
        # 自注意力机制，q,k,v都是自己
        q = k = v = x
        attention_output = self.attention_model(q,k,v)
        # 归一化 + 残差连接
        attention_norm_output = self.atten_norm_model(attention_output + x)
        # 前馈
        ff_output = self.ff_model(attention_norm_output)
        ff_norm_output = self.ff_norm_model(ff_output + attention_norm_output)

        output = self.dropout(ff_norm_output)
        
        return output




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
        pos_embedded = self.pos_embedded(embedded)
        temp_src = pos_embedded
        # 多个编码块
        for block in self.encoder_blocks:
            temp_src = block(temp_src)
        return temp_src


if __name__ == '__main__':
    # test embedding
    pairs, test_pairs, src_vocab, tgt_vocab = get_dataset(min_len=0, max_len=5)
    
    encoder = TransformerEncoder(src_vocab, tgt_vocab)
    
    
    

    epoch_batches = TensorHandler.pairs_to_batches(pairs, 5)
    for src_tensor, tgt_input_tensor, tgt_output_tensor in epoch_batches:
        encoder_output = encoder(src_tensor)


        break
        
        

    

    


    print(f"test in transformer input layer")
    pass
