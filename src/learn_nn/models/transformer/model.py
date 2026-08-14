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
    

    def _teaching_force(self, tgt_input, encoder_output):
        logits = self.decoder(tgt_input, encoder_output, encoder_output)
        return logits

    def _self_generation(self, input_token, encoder_output):
        logits = []
        _max_length = 50
        while True:
            logit = self.decoder(input_token, encoder_output, encoder_output)
            # 只取最后一个时间步的 logit: [T, B, V] -> [1, B, V]
            next_logit = logit[-1:, :, :]
            logits.append(next_logit)
            # 取最后一步的 token 并追加到 input_token 末尾
            next_token = torch.argmax(next_logit, dim=-1)  # [1, B]
            input_token = torch.cat([input_token, next_token], dim=0)
            # 所有 batch 都生成了 EOS 则停止
            if next_token.eq(EOS_IDX).all():
                break
            if len(logits) >= _max_length:
                break
        # [T_gen, B, V]
        logits = torch.cat(logits, dim=0)
        return logits




    def forward(self, src, tgt_input=None):
        encoder_output = self.encoder(src)
        
        logits = None
        if tgt_input is None:
            batch_size = src.size(1)
            input_token = torch.full(
                (1, batch_size),
                fill_value=SOS_IDX,
                dtype=torch.long,
            )            
            logits = self._self_generation(input_token, encoder_output)
        else:
            logits = self._teaching_force(tgt_input, encoder_output)
        return logits









if __name__ == '__main__':
    print('model: ', __package__)
    print(f"Q,K,V: ", query, key, value)
    pass