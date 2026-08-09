import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from learn_nn.train_datasets.get_tatoeba import EOS_IDX, PAD_IDX, SOS_IDX


SRC_EMBED_SIZE = 256
EMBED_SIZE = 512

class Encoder(nn.Module):
    def __init__(self, src_vocab):
        super().__init__()
        self.embedding = nn.Embedding(
            len(src_vocab),
            SRC_EMBED_SIZE,
            padding_idx=PAD_IDX,
        )
        self.rnn = nn.GRU(SRC_EMBED_SIZE, EMBED_SIZE, batch_first=False)

    def forward(self, src):
        lengths = src.ne(PAD_IDX).sum(dim=0).cpu()
        embedded = self.embedding(src)
        packed = pack_padded_sequence(
            embedded,
            lengths,
            enforce_sorted=False,
        )
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            total_length=src.size(0),
        )
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, tgt_vocab, use_attention: bool = False):
        super().__init__()
        _input_size = EMBED_SIZE
        _hidden_size = EMBED_SIZE
        self.use_attention = use_attention
        self.embedding = nn.Embedding(
            len(tgt_vocab),
            _input_size,
            padding_idx=PAD_IDX,
        )
        self.rnn = nn.GRU(_input_size, _hidden_size, batch_first=False)
        self.attention_proj = nn.Linear(2 * _hidden_size, _hidden_size)
        self.linear = nn.Linear(_hidden_size, len(tgt_vocab))
        self.linear.weight = self.embedding.weight
    
     # 输入Q和K，计算出V
    # 探索使用不同的数据作为Q和K，观测一下效果
    def get_attention_hidden(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 输出的V，形状跟Query一致
        # V的形状应该对齐Decoder的Hidden形状，即[T, B, H]
        # 计算Score，是Query跟Key的点积(并开平方)，
        # Q: 查询输入，暂定embedded，形状是[T, B, H],T是tgt_input序列长度
        # K：Encoder的每一步输出，形状是[S, B, H],S是src_input的序列长度
        
        # query跟key的相关性分数
        score = torch.einsum(
            "tbh,sbh->bts",
            query,
            key
        ) / (query.size(-1) ** 0.5)
        score = score.masked_fill(
            ~src_mask[:, None, :],
            torch.finfo(score.dtype).min,
        )

        attention_weights = torch.softmax(score, dim=-1)
        
        # 相关性分数跟key做点积，算出context，形状对齐query
        context = torch.einsum(
            "bts,sbh->tbh",
            attention_weights,
            key,
        )
        output = torch.tanh(
            self.attention_proj(
                torch.cat([query, context], dim=-1)
            )
        )

        return output, output[-1:]


    def forward(self, tgt_input, previous_hidden, encoder_outputs, src_mask):
        # Embedded Size: (T, B, H).
        # T: padding 后的长度，在RNN场景中是可以不固定的
        # B: Batch Size， 在RNN场景中也是可以不固定的
        # H: Embedding Size, 这个尺寸要一开始设置好，跟词表规模对应
        embedded = self.embedding(tgt_input)
        if self.use_attention:
            previous_hidden, _ = self.get_attention_hidden(
                previous_hidden,
                encoder_outputs,
                src_mask,
            )
        _hidden = previous_hidden
        
        decoder_output, decoder_hidden = self.rnn(embedded, _hidden)

        if self.use_attention:
            decoder_output, decoder_hidden = self.get_attention_hidden(
                decoder_output,
                encoder_outputs,
                src_mask,
            )
        
        # previous_hidden的形状是1, B, H
        
        logits = self.linear(decoder_output)
        return logits, decoder_hidden

class Seq2SeqModel(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, use_attention: bool = False):
        super().__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.encoder: Encoder = Encoder(src_vocab)
        self.decoder: Decoder = Decoder(tgt_vocab, use_attention=use_attention)
        

    def forward(self, src, tgt_input):
        outputs, hidden = self.encoder(src)
        src_mask = src.transpose(0, 1).ne(PAD_IDX)
        logits, _ = self.decoder(tgt_input, hidden, outputs, src_mask)
        return logits

    def inference(self, src, max_len):
        outputs, hidden = self.encoder(src)
        src_mask = src.transpose(0, 1).ne(PAD_IDX)
        logits_steps = []
        batch_size = src.size(1)
        # 所有推理都从SOS_IDX开始
        input_token = torch.full(
            (1, batch_size),
            fill_value=SOS_IDX,
            dtype=torch.long,
            device=src.device,
        )
        
        while True:
            logits, hidden = self.decoder(
                input_token,
                hidden,
                outputs,
                src_mask,
            )
            logits_steps.append(logits)
            input_token = logits.argmax(dim=-1)
            if input_token.eq(EOS_IDX).all():
                break
            if len(logits_steps) >= max_len:
                break

        return torch.cat(logits_steps, dim=0)

        

def main():
    print("hello world _model_")

if __name__ == "__main__":
    main()
