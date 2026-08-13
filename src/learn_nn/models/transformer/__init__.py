
from learn_nn.train_framework import TrainWorker, EvaluateWorker, get_dataset, log_output_line
from learn_nn.models.transformer.model import TimTransformer

import random
import torch
import numpy as np
_SEED_ = 88
random.seed(_SEED_)
torch.manual_seed(_SEED_)
torch.cuda.manual_seed_all(_SEED_)
np.random.seed(_SEED_)

EPOCHS = 100
DATA_TOKEN_MAX_LEN = 512


def main():
    pairs, test_pairs, src_vocab, tgt_vocab = get_dataset(min_len=0, max_len=DATA_TOKEN_MAX_LEN)

    transformer_model = TimTransformer(src_vocab, tgt_vocab)
    transformer_trainer = TrainWorker(transformer_model, EPOCHS)
    transformer_trainer.train(pairs)

    transformer_model.eval()

    transformer_evaluator = EvaluateWorker(transformer_model, "transformer_3layer")
    transformer_avg_chrf, _ = transformer_evaluator.test(test_pairs)

    log_output_line(f"\n=== 完成 ===")
    log_output_line(f"  训练集大小: {len(pairs)}")
    log_output_line(f"  测试集大小: {len(test_pairs)}")
    log_output_line(f"  Transformer (3-layer) avg chrF: {transformer_avg_chrf:.4f}")


if __name__ == "__main__":
    main()
