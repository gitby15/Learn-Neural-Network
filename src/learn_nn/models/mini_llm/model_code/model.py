import torch
import torch.nn as nn
import torch.nn.functional as F
from learn_nn.models.mini_llm.model_code.base_component import Embedding, PosEmbedding, TransformerBlock, RMSNorm, Linear


class MiniLLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = Embedding(vocab_size)
        self.pos_embedding = PosEmbedding()
        self.embedding_dropout = nn.Dropout()

        self.attention_blocks = nn.ModuleList([
            TransformerBlock() for _ in range(6)
        ])

        self.rms_norm = RMSNorm()
        self.linear = Linear(vocab_size)

    # [B, T, C]
    def forward(self, x):
        embed_output = self.embedding(x) 
        pos_embed_output = self.pos_embedding(embed_output)
        pos_embed_output = self.embedding_dropout(pos_embed_output)

        attention_output = pos_embed_output
        for block in self.attention_blocks:
            attention_output = block(attention_output, True)

        norm_output = self.rms_norm(attention_output)
        logits = self.linear(norm_output)
        logits = F.softmax(logits, dim=-1)
        return logits # [B, T, V]


if __name__ == '__main__':
    model = MiniLLM()
    # [B, T]
    x = torch.ones(4,5,dtype=torch.long)
    output = model(x)
    print(output)