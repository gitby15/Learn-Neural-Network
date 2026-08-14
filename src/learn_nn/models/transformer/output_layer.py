import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from learn_nn.models.transformer.base_component import Embeddings, FinalOutput, PositionEncoding, MultiHeadAttention, FeedForword, LayerNorm



class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.masked_multi_head_attention_model = MultiHeadAttention()
        self.masked_atten_norm_model = LayerNorm()
        self.cross_attention_model = MultiHeadAttention()
        self.atten_norm_model = LayerNorm()
        self.ff_model = FeedForword()
        self.ff_norm_model = LayerNorm()
    def forward(self, decoder_input, encoder_k, encoder_v, causal_mask=None):
        # Todo: 这里的QKV还是自己吗？
        q = k = v = decoder_input

        # 对输入进行掩码自注意力
        masked_attention_output = self.masked_multi_head_attention_model(q,k,v, causal_mask)
        # 残差连接 + 归一化
        masked_attention_norm_output = self.masked_atten_norm_model(masked_attention_output + decoder_input)

        # 结合Decoder的输入和Encoder的输出计算注意力
        cross_attention_output = self.cross_attention_model(masked_attention_norm_output, encoder_k, encoder_v)
        # 残差连接 + 归一化
        cross_attention_norm_output = self.atten_norm_model(cross_attention_output + masked_attention_norm_output)

        # 前馈
        ff_output = self.ff_model(cross_attention_norm_output)
        ff_norm_output = self.ff_norm_model(ff_output + cross_attention_norm_output)
        return ff_norm_output


class TransformerDecoder(nn.Module):
    def __init__(self, src_vocab, tgt_vocab):
        super().__init__()
        self.embeddings = Embeddings(tgt_vocab)
        self.pos_encoding = PositionEncoding()
        self.decoder_blocks = nn.ModuleList([DecoderBlock() for _ in range(6)])
        
        self.final_output = FinalOutput(src_vocab, tgt_vocab)
    
    def forward(self, tgt_input, encoder_k, encoder_v):
        embedded = self.embeddings(tgt_input)
        embedded = self.pos_encoding(embedded)
        output = embedded

        
        causal_mask = self.generate_causal_mask(tgt_input)

        for block in self.decoder_blocks:
            output = block(output, encoder_k, encoder_v, causal_mask=causal_mask)
        output = self.final_output(output)
        return output

    def generate_causal_mask(self, tgt_input):
        # Todo: 画个图研究一下这个是怎么个三角法
        # Todo: 还是有很多地方需要画图的，研究一下可观测性在无UI服务器怎么搞
        tgt_len = tgt_input.size(0)
        mask = torch.tril(torch.ones(tgt_len, tgt_len)) == 1
        return mask



if __name__ == "__main__":
    from learn_nn.models.transformer.input_layer import TransformerEncoder
    pairs, test_pairs, src_vocab, tgt_vocab = get_dataset(min_len=0, max_len=5)
    encoder = TransformerEncoder(src_vocab, tgt_vocab)
    decoder = TransformerDecoder(src_vocab, tgt_vocab)
    
    
    

    epoch_batches = TensorHandler.pairs_to_batches(pairs, 5)
    for src_tensor, tgt_input_tensor, tgt_output_tensor in epoch_batches:
        encoder_output = encoder(src_tensor)
        decoder_output = decoder(tgt_input_tensor, encoder_output, encoder_output)

        break
        
        

        