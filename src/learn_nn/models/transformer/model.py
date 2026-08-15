import torch
import torch.nn as nn
import torch.nn.functional as F
from learn_nn.models.transformer.input_layer import TransformerEncoder
from learn_nn.models.transformer.output_layer import TransformerDecoder
from learn_nn.train_framework.dataset.get_tatoeba import EOS_IDX, PAD_IDX, SOS_IDX


class TimTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab, tgt_vocab)
        self.decoder = TransformerDecoder(src_vocab, tgt_vocab)
    

    def _teaching_force(self, tgt_input, encoder_output, src_padding_mask):
        logits = self.decoder(tgt_input, encoder_output, encoder_output, src_padding_mask=src_padding_mask)
        return logits

    def _self_generation(self, input_token, encoder_output, src_padding_mask):
        logits = []
        _max_length = 50
        while True:
            decoder_output = self.decoder(
                input_token, encoder_output, encoder_output, src_padding_mask=src_padding_mask
            )
            next_logit = decoder_output[:, -1:, :]  # [B, T, V] -> [B, 1, V]
            logits.append(next_logit)
            # 取最后一步的 token 并追加到 input_token 末尾
            next_token = torch.argmax(next_logit, dim=-1)  # [B, 1]
            input_token = torch.cat([input_token, next_token], dim=1)
            # 所有 batch 都生成了 EOS 则停止
            if next_token.eq(EOS_IDX).all():
                break
            if len(logits) >= _max_length:
                break
        logits = torch.cat(logits, dim=1)  # list[B,1,V] -> [B, T_gen, V]
        return logits

    def forward(self, src, tgt_input=None):
        encoder_output, src_padding_mask = self.encoder(src)
        
        logits = None
        if tgt_input is not None:
            logits = self._teaching_force(tgt_input, encoder_output, src_padding_mask)
        else:
            batch_size = src.size(0)
            input_token = torch.full(
                (batch_size, 1),
                fill_value=SOS_IDX,
                dtype=torch.long,
                device=src.device,
            )            
            logits = self._self_generation(input_token, encoder_output, src_padding_mask)
            
        return logits









if __name__ == '__main__':
    print(f'model module: {__package__}')