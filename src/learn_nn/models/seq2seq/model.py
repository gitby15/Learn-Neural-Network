import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Todo: 从train_framework中引入EOS_IDX, PAD_IDX, SOS_IDX，或者有更好的架构，不让model直接引入它们
from learn_nn.train_framework.dataset.get_tatoeba import EOS_IDX, PAD_IDX, SOS_IDX


def _calculate_src_embed_size(src_vocab):
    _len = len(src_vocab)
    if _len < 5000:
        return 128
    elif _len < 10000:
        return 256
    return 512

def _calculate_tgt_embed_size(tgt_vocab):
    _len = len(tgt_vocab)
    if _len < 5000:
        return 128
    elif _len < 10000:
        return 256
    return 512


class Encoder(nn.Module):
    def __init__(self, src_vocab, tgt_vocab):
        super().__init__()
        _src_embed_size = _calculate_src_embed_size(src_vocab)
        _tgt_embed_size = _calculate_tgt_embed_size(tgt_vocab)
        _dropout_rate = 0.1
        self.embedding = nn.Embedding(
            len(src_vocab),
            _src_embed_size,
            padding_idx=PAD_IDX,
        )
        self.rnn = nn.GRU(_src_embed_size, _tgt_embed_size, batch_first=True)
        self.dropout = nn.Dropout(_dropout_rate)

    def forward(self, src):
        # src: [B, T]
        lengths = src.ne(PAD_IDX).sum(dim=1).cpu()
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        packed = pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=src.size(1),
        )
        # outputs: [B, T, H], hidden: GRU原生[1,B,H] -> 对外统一BTC: [B,1,H]
        hidden = hidden.transpose(0, 1).contiguous()
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, tgt_vocab, use_attention: bool = False):
        super().__init__()
        _input_size = _calculate_tgt_embed_size(tgt_vocab)
        _hidden_size = _input_size
        self.use_attention = use_attention
        self.embedding = nn.Embedding(
            len(tgt_vocab),
            _input_size,
            padding_idx=PAD_IDX,
        )

        if self.use_attention:
            self.attention_proj = nn.Linear(2 * _hidden_size, _hidden_size)

        self.rnn = nn.GRU(_input_size, _hidden_size, batch_first=True)
        self.linear = nn.Linear(_hidden_size, len(tgt_vocab))
        self.linear.weight = self.embedding.weight
    
    # 输入Q和K，计算出V
    # 探索使用不同的数据作为Q和K，观测一下效果
    def get_attention_context(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 输出的V，形状跟Query一致
        # V的形状应该对齐Decoder的Hidden形状，即[B, T, H]
        # 计算Score，是Query跟Key的点积(并开平方)，
        # Q: 查询输入，形状是[B, T, H],T是tgt_input序列长度
        # K：Encoder的每一步输出，形状是[B,S , H],S是src_input的序列长度
        
        # query跟key的相关性分数
        score = torch.einsum(
            "bth,bsh->bts",
            query,
            key
        ) / (query.size(-1) ** 0.5)
        score = score.masked_fill(
            ~src_mask[:, None, :],
            torch.finfo(score.dtype).min,
        )

        attention_weights = torch.softmax(score, dim=-1)

        context = torch.einsum(
            "bts,bsh->bth",
            attention_weights,
            key,
        )
        return context


    def forward(self, tgt_input, previous_hidden, encoder_outputs, src_mask):
        # tgt_input: [B, T]
        # previous_hidden: [B, 1, H] (BTC)
        # encoder_outputs: [B, T_s, H]
        # src_mask: [B, T_s]
        embedded = self.embedding(tgt_input)
        embedded = F.relu(embedded)

        # GRU内部要求hidden为 [num_layers, B, H]，将BTC [B,1,H] 转回
        _hidden = previous_hidden.transpose(0, 1).contiguous()

        decoder_output, decoder_hidden = self.rnn(embedded, _hidden)
        # decoder_output: [B, T, H], decoder_hidden(GRU原生): [1, B, H] -> 转回BTC [B,1,H]
        decoder_hidden = decoder_hidden.transpose(0, 1).contiguous()

        if self.use_attention:
            attention = self.get_attention_context(decoder_output, encoder_outputs, src_mask)
            decoder_output = torch.tanh(
                self.attention_proj(torch.cat((decoder_output, attention), dim=-1))
            )

        logits = self.linear(decoder_output)
        # logits: [B, T, V]
        return logits, decoder_hidden


class Seq2SeqModel(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, use_attention: bool = False):
        super().__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.encoder: Encoder = Encoder(src_vocab, tgt_vocab)
        self.decoder: Decoder = Decoder(tgt_vocab, use_attention=use_attention)


    def forward_step(self, input_token, hidden, encoder_outputs, src_mask):
        logits, hidden = self.decoder(
            input_token,
            hidden,
            encoder_outputs,
            src_mask,
        )
        return logits, hidden

    def _teaching_force(self, encoder_outputs, hidden, src_mask, tgt_input):
        logits, _ = self.decoder(tgt_input, hidden, encoder_outputs, src_mask)
        return logits

    def _self_generation(self, encoder_outputs, hidden, src_mask, input_token):
        logits_steps = []
        _max_length = encoder_outputs.size(1) * 3

        while True:
            # GRU推理: 每步只喂1个token [B,1], hidden携带历史
            logits, hidden = self.forward_step(input_token, hidden, encoder_outputs, src_mask)
            logits_steps.append(logits)
            next_token = torch.argmax(logits[:, -1:, :], dim=-1)  # [B,1]
            input_token = next_token  # 下一步只喂新token

            if next_token.eq(EOS_IDX).all():
                break
            if len(logits_steps) >= _max_length:
                break
        return torch.cat(logits_steps, dim=1)  # list[B,1,V] -> [B, T_gen, V]

    def forward(self, src, tgt_input=None):
        # src: [B, T_src]
        encoder_outputs, encoder_hidden = self.encoder(src)
        src_mask = src.ne(PAD_IDX)  # [B, T_src]
        batch_size = src.size(0)

        if tgt_input is not None:
            logits = self._teaching_force(encoder_outputs, encoder_hidden, src_mask, tgt_input)
        else:
            input_token = torch.full(
                (batch_size, 1),
                fill_value=SOS_IDX,
                dtype=torch.long,
                device=src.device,
            )
            logits = self._self_generation(encoder_outputs, encoder_hidden, src_mask, input_token)
        return logits


def main():
    print("hello world _model_")

if __name__ == "__main__":
    main()
