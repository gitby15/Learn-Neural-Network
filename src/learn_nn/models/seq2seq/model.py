import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Todo: 从train_framework中引入EOS_IDX, PAD_IDX, SOS_IDX，或者有更好的架构，不让model直接引入它们
from learn_nn.train_framework.dataset.get_tatoeba import EOS_IDX, PAD_IDX, SOS_IDX


SRC_EMBED_SIZE = 128
EMBED_SIZE = 64

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
        
        if self.use_attention:
            self.attention_proj = nn.Linear(2 * _hidden_size, _hidden_size)
        
        self.rnn = nn.GRU(_input_size, _hidden_size, batch_first=False)
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
        # V的形状应该对齐Decoder的Hidden形状，即[T, B, H]
        # 计算Score，是Query跟Key的点积(并开平方)，
        # Q: 查询输入，形状是[T, B, H],T是tgt_input序列长度
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
        return context


    def forward(self, tgt_input, previous_hidden, encoder_outputs, src_mask):
        # Embedded Size: (T, B, H).
        # T: padding 后的长度，在RNN场景中是可以不固定的
        # B: Batch Size， 在RNN场景中也是可以不固定的
        # H: Embedding Size, 这个尺寸要一开始设置好，跟词表规模对应
        embedded = self.embedding(tgt_input)

        _hidden = previous_hidden

        decoder_output, decoder_hidden = self.rnn(embedded, _hidden)

        if self.use_attention:
            attention = self.get_attention_context(decoder_output, encoder_outputs, src_mask)
            decoder_output = torch.tanh(
                self.attention_proj(torch.cat((decoder_output, attention), dim=-1))
            )
            decoder_hidden = decoder_output[-1:]

        logits = self.linear(decoder_output)
        return logits, decoder_hidden

class Seq2SeqModel(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, use_attention: bool = False):
        super().__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.encoder: Encoder = Encoder(src_vocab)
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
        _max_length = encoder_outputs.size(0) * 3
            
        while True:
            logits, hidden = self.forward_step(input_token, hidden, encoder_outputs, src_mask)
            logits_steps.append(logits)
            input_token = torch.argmax(logits, dim=-1)

            if input_token.eq(EOS_IDX).all():
                break
            if len(logits_steps) >= _max_length:
                break
        return torch.cat(logits_steps, dim=0)

    # tgt_input 有值，说明要做teaching forcing
    def forward(self, src, tgt_input = None):
        encoder_outputs, encoder_hidden = self.encoder(src)
        src_mask = src.transpose(0, 1).ne(PAD_IDX)
        batch_size = src.size(1)
        logits = []

        if tgt_input is not None:
            logits = self._teaching_force(encoder_outputs, encoder_hidden, src_mask, tgt_input)
        else:
            input_token = torch.full(
                (1, batch_size),
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
