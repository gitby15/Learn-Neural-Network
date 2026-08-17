import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from learn_nn.models.mini_llm.model_code.base_component import AttentionLayer, ATTENTION_LAYER_COUNT, DROPOUT_RATE, EMBEDDING_SIZE, Embedding, RMSNorm, Linear


class MiniLLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = Embedding(vocab_size)
        self.embedding_dropout = nn.Dropout(DROPOUT_RATE)

        self.attention_layer = AttentionLayer()

        self.rms_norm = RMSNorm()
        self.linear = Linear(vocab_size)
        self.linear.linear.weight = self.embedding.embedding.weight

        self._init_weights()

    def _init_weights(self):
        init_std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)

        for pn, p in self.named_parameters():
            if pn.endswith("wo.weight") or pn.endswith("w_o.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=init_std / math.sqrt(2 * ATTENTION_LAYER_COUNT)
                )

    def forward(self, x):
        embed_output = self.embedding(x)
        embed_output = self.embedding_dropout(embed_output)
        attention_output = self.attention_layer(embed_output)
        norm_output = self.rms_norm(attention_output)
        logits = self.linear(norm_output)
        return logits


if __name__ == '__main__':
    model = MiniLLM(vocab_size=6400)
    x = torch.ones(4,5,dtype=torch.long)
    output = model(x)
    print(output.shape)
