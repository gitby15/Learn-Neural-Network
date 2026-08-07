
from learn_nn.models.seq2seq._train_ import TrainWorker, EvaluateWorker, TensorHandler, chrf_score
from learn_nn.models.seq2seq._model_ import Seq2SeqModel
from learn_nn.train_datasets.get_tatoeba import get_dataset
from learn_nn.models._utils_ import (
    log_output_line,
    save_model,
)


def main():
    pairs, test_pairs, src_vocab, tgt_vocab = get_dataset(min_len=0, max_len=5)
    model = Seq2SeqModel(src_vocab, tgt_vocab)



    # ===== 1) 训练：输入字符串 pairs =====
    trainer = TrainWorker(model)
    trainer.train(pairs)          # 要求1：train 接收字符串pairs

    # ===== 2) 保存模型 =====
    model.eval()
    save_model(model)

    # ===== 3) 评估：逐行推理 + chrF 评分 =====
    evaluator = EvaluateWorker(model)
    avg_chrf, _ = evaluator.test(test_pairs)   # 要求1/2/4：字符串pairs输入 / 逐行 / chrF

    print(f"\n=== 完成 ===")
    print(f"  训练集大小: {len(pairs)}")
    print(f"  测试集大小: {len(test_pairs)}")
    print(f"  测试集avg chrF: {avg_chrf:.4f}")


if __name__ == "__main__":
    main()
