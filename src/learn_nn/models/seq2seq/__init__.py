
from learn_nn.train_framework import TrainWorker, EvaluateWorker, get_dataset,log_output_line
from learn_nn.models.seq2seq.model import Seq2SeqModel

import random
import torch
import numpy as np
_SEED_ = 88
random.seed(_SEED_)
torch.manual_seed(_SEED_)
torch.cuda.manual_seed_all(_SEED_)
np.random.seed(_SEED_)

EPOCHS = 50
DATA_TOKEN_MAX_LEN = 6


def main():
    pairs, test_pairs, src_vocab, tgt_vocab = get_dataset(min_len=0, max_len=DATA_TOKEN_MAX_LEN)
    
    attention_model = Seq2SeqModel(src_vocab, tgt_vocab, use_attention=True)
    attention_trainer = TrainWorker(attention_model, EPOCHS)
    attention_trainer.train(pairs)

    non_attention_model = Seq2SeqModel(src_vocab, tgt_vocab, use_attention=False)
    non_attention_traniner = TrainWorker(non_attention_model, EPOCHS)
    non_attention_traniner.train(pairs)

    # 训练完了，冻结模型
    attention_model.eval()
    non_attention_model.eval()

    
    attention_evaluator = EvaluateWorker(attention_model, "attention")
    attention_avg_chrf, _ = attention_evaluator.test(test_pairs)   # 要求1/2/4：字符串pairs输入 / 逐行 / chrF

    non_attention_evaluator = EvaluateWorker(non_attention_model, "non_attention")
    non_attention_avg_chrf, _ = non_attention_evaluator.test(test_pairs)   # 要求1/2/4：字符串pairs输入 / 逐行 / chrF

    log_output_line(f"\n=== 完成 ===")
    log_output_line(f"  训练集大小: {len(pairs)}")
    log_output_line(f"  测试集大小: {len(test_pairs)}")
    log_output_line(f"  测试集attention avg chrF: {attention_avg_chrf:.4f}, non-attention avg chrF: {non_attention_avg_chrf:.4f}")


if __name__ == "__main__":
    main()
