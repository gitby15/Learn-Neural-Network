import torch
from torch import nn
from learn_nn.train_datasets.get_tatoeba import EOS_IDX,SOS_IDX


SRC_EMBED_SIZE = 128
EMBED_SIZE = 256

class Encoder(nn.Module):
    def __init__(self, src_vocab):
        super().__init__()
        self.embedding = nn.Embedding(len(src_vocab), SRC_EMBED_SIZE)
        self.rnn = nn.GRU(SRC_EMBED_SIZE, EMBED_SIZE, batch_first=False)

    def forward(self, src):
        embedded = self.embedding(src)
        # 这里没有输出output，只输出隐藏状态
        # 因为解码器需要上一个时间步的隐藏状态作为输入
        _, hidden = self.rnn(embedded)
        return hidden


class ReserveEncoder(nn.Module):
    def __init__(self, src_vocab):
        super().__init__()
        self.embedding = nn.Embedding(len(src_vocab), SRC_EMBED_SIZE)
        # 双向 2 层 GRU：每个位置都能同时看过去和未来的上下文
        # 注：为了简化，保持 num_layers=1、bidirectional=True，后续加层再扩
        self.rnn = nn.GRU(
            SRC_EMBED_SIZE,
            EMBED_SIZE,
            num_layers=1,
            bidirectional=True,
            batch_first=False,
        )
        # 双向 GRU 返回的 hidden 形状是 [2*num_layers, B, H]
        # 把 2*H 投影回 H，保证 Decoder 侧接口不变
        self.hidden_proj = nn.Linear(2 * EMBED_SIZE, EMBED_SIZE)

    def forward(self, src):
        embedded = self.embedding(src)
        # outputs: [T, B, 2*H]   每个时刻的前/后向 hidden 拼接
        # hidden:  [2*num_layers, B, H]
        outputs, hidden = self.rnn(embedded)
        # hidden 的 0 号 dim 顺序是 [layer0_fwd, layer0_bwd]
        # 把同一层两个方向沿最后一维拼接 -> [B, 2*H]，再投影 -> [B, H]
        fwd = hidden[0]   # [B, H]
        bwd = hidden[1]   # [B, H]
        merged = torch.cat([fwd, bwd], dim=-1)   # [B, 2H]
        hidden = torch.tanh(self.hidden_proj(merged)).unsqueeze(0)   # [1, B, H]
        return hidden

class Decoder(nn.Module):
    def __init__(self, tgt_vocab):
        super().__init__()
        self.embedding = nn.Embedding(len(tgt_vocab), EMBED_SIZE)
        self.rnn = nn.GRU(EMBED_SIZE, EMBED_SIZE, batch_first=False)
        self.linear = nn.Linear(EMBED_SIZE, len(tgt_vocab))
        self.linear.weight = self.embedding.weight
    
    def forward(self, tgt_input, hidden):
        embedded = self.embedding(tgt_input)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.linear(output)
        return logits, hidden

class Seq2SeqModel(nn.Module):
    def __init__(self, src_vocab, tgt_vocab):
        super().__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.encoder: Encoder = Encoder(src_vocab)
        # self.encoder: Encoder = ReserveEncoder(src_vocab)
        self.decoder: Decoder = Decoder(tgt_vocab)
        

    def forward(self, src, tgt_input):
        hidden = self.encoder(src)
        logits, _ = self.decoder(tgt_input, hidden)
        return logits

    def inference(self, src, max_len):
        hidden = self.encoder(src)
        logits_steps = []
        batch_size = src.size(1)
        input_token = torch.full(
            (1, batch_size),
            fill_value=SOS_IDX,
            dtype=torch.long,
            device=src.device,
        )
        
        while True:
            logits, hidden = self.decoder(input_token, hidden)
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
