import random

import torch
from torch import nn
from tqdm import tqdm
from learn_nn.models._utils_ import get_device
from learn_nn.train_datasets.get_tatoeba import (
    EOS_IDX, PAD_IDX, SOS_IDX, UNK_IDX,
    get_dataset,
    tokenize_source, tokenize_target,
    idx_list_to_token_source, idx_list_to_token_target,
)
from learn_nn.models._utils_ import (
    log_output_line, save_model,
    INFERENCE_START_TAG, INFERENCE_END_TAG,
)

BATCH_SIZE = 32


def chrf_score(
    pred_idx_list: list[int],
    ref_idx_list: list[int],
    max_order: int = 6,
    beta: float = 2.0,
) -> float:
    """字符级 n-gram F-score（chrF）。输入是预测/参考的 token id list。"""

    def normalize_target_idx_list(idx_list: list[int]) -> list[int]:
        special = {PAD_IDX, SOS_IDX, EOS_IDX}
        return [idx for idx in idx_list if idx not in special]


    def target_idx_to_text(idx_list: list[int]) -> str:
        return idx_list_to_token_target(normalize_target_idx_list(idx_list))

    pred_text = target_idx_to_text(pred_idx_list)
    ref_text = target_idx_to_text(ref_idx_list)

    if not pred_text and not ref_text:
        return 1.0
    if not pred_text or not ref_text:
        return 0.0

    pred_chars = list(pred_text)
    ref_chars = list(ref_text)

    def _ngram_counts(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
        counts: dict[tuple[str, ...], int] = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            counts[ngram] = counts.get(ngram, 0) + 1
        return counts

    precision_sum = 0.0
    recall_sum = 0.0
    valid_order_count = 0
    for order in range(1, max_order + 1):
        if len(pred_chars) < order or len(ref_chars) < order:
            continue
        pred_counts = _ngram_counts(pred_chars, order)
        ref_counts = _ngram_counts(ref_chars, order)
        overlap = sum(min(cnt, ref_counts.get(ng, 0)) for ng, cnt in pred_counts.items())
        pred_total = sum(pred_counts.values())
        ref_total = sum(ref_counts.values())
        precision_sum += overlap / pred_total if pred_total else 0.0
        recall_sum += overlap / ref_total if ref_total else 0.0
        valid_order_count += 1

    if valid_order_count == 0:
        return 0.0

    avg_p = precision_sum / valid_order_count
    avg_r = recall_sum / valid_order_count
    if avg_p == 0.0 and avg_r == 0.0:
        return 0.0

    beta_sq = beta * beta
    return (1 + beta_sq) * avg_p * avg_r / (beta_sq * avg_p + avg_r)

class TensorHandler:

    # --------------------------------------------------------
    # Level 2: idx_list → 单条 Tensor（给 EvaluateWorker 逐行推理用）
    # --------------------------------------------------------
    @staticmethod
    def src_idx_to_train_tensor(src_idx_list: list[int]) -> torch.Tensor:
        """给单条推理用: [SOS] + idx + [EOS] → 形状 [T, 1]"""
        seq = [SOS_IDX] + list(src_idx_list) + [EOS_IDX]
        tensor = torch.tensor(seq, dtype=torch.long, device=get_device())
        return tensor.unsqueeze(1)   # [T] → [T, 1]

    @staticmethod
    def tgt_idx_to_input_tensor(tgt_idx_list: list[int]) -> torch.Tensor:
        """（EvaluateWorker 不需要 teacher forcing, 但保留给调试用）"""
        seq = [SOS_IDX] + list(tgt_idx_list)
        tensor = torch.tensor(seq, dtype=torch.long, device=get_device())
        return tensor.unsqueeze(1)

    # --------------------------------------------------------
    # Level 3: idx_pair_batch → 批量 Tensor（给 TrainWorker 用）
    # --------------------------------------------------------
    @staticmethod
    def _pad_sequences(sequences: list[list[int]]) -> torch.Tensor:
        max_len = max(len(s) for s in sequences)
        padded: list[list[int]] = []
        for s in sequences:
            ids = list(s)
            ids += [PAD_IDX] * (max_len - len(ids))
            padded.append(ids)
        tensor = torch.tensor(padded, dtype=torch.long, device=get_device())
        return tensor.transpose(0, 1)   # [B, T] → [T, B]

    @staticmethod
    def idx_pair_to_batch_tensors(
        idx_pair_batch: list[tuple[list[int], list[int]]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_seqs: list[list[int]] = []
        tgt_in_seqs: list[list[int]] = []
        tgt_out_seqs: list[list[int]] = []
        for src_idx, tgt_idx in idx_pair_batch:
            src_seqs.append([SOS_IDX] + list(src_idx) + [EOS_IDX])
            tgt_in_seqs.append([SOS_IDX] + list(tgt_idx))
            tgt_out_seqs.append(list(tgt_idx) + [EOS_IDX])
        return (
            TensorHandler._pad_sequences(src_seqs),
            TensorHandler._pad_sequences(tgt_in_seqs),
            TensorHandler._pad_sequences(tgt_out_seqs),
        )

    # --------------------------------------------------------
    # 完整入口: 字符串 pairs → 所有训练 batches（给 TrainWorker 用）
    # --------------------------------------------------------
    @staticmethod
    def pairs_to_batches(
        train_pairs: list[tuple[str, str]],
        batch_size: int = BATCH_SIZE,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        输入: list[(src_str, tgt_str)]
        输出: list[(src_tensor, tgt_in_tensor, tgt_out_tensor)]
        """
        # 1) str → idx
        idx_pairs: list[tuple[list[int], list[int]]] = []
        for s, t in train_pairs:
            idx_pairs.append((tokenize_source(s), tokenize_target(t)))

        # 2) 切 batch
        idx_batches = [
            idx_pairs[i:i + batch_size]
            for i in range(0, len(idx_pairs), batch_size)
        ]

        # 3) 每个 batch → tensor
        result = []
        for ib in tqdm(idx_batches, desc="Preparing batches"):
            result.append(TensorHandler.idx_pair_to_batch_tensors(ib))
        return result


# ============================================================
# TrainWorker
# 要求1: 输入的 pair 都是字符串
# ============================================================
class TrainWorker:
    def __init__(self, model: nn.Module):
        self.model = model
        _device = get_device()
        self.model.to(_device)
        self.epochs: int = 20

    def train(self, train_pairs: list[tuple[str, str]]) -> None:
        """
        要求1: train_pairs 每个元素都是 (src_str, tgt_str), 纯字符串
        """
       
        _model = self.model
        batches = TensorHandler.pairs_to_batches(train_pairs)
        epoch_batches = list(batches)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.05)
        optimizer = torch.optim.Adam(_model.parameters(), lr=0.001)
        progress = tqdm(range(self.epochs), desc="Training")

         # ===== 打印元信息 =====
        log_output_line("=========== Train Meta: ===========")
        log_output_line(f"MODEL_STRUCTURE: {repr(_model)}")
        log_output_line(
            f"TRAINING_PAIRS_LEN: {len(epoch_batches)} | "
            f"EPOCHS: {self.epochs}"
        )
        log_output_line("=========== Train Meta End: ===========")
        for idx in progress:
            _model.train()
            epoch_loss = 0.0
            random.shuffle(epoch_batches)
            for src_tensor, tgt_input_tensor, tgt_output_tensor in epoch_batches:
                logits = _model(src_tensor, tgt_input_tensor)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_output_tensor.reshape(-1),
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(_model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg = epoch_loss / max(1, len(epoch_batches))
            progress.set_postfix(loss=f"{avg:.6f}")
            log_output_line(f"Idx: {idx} | Training Loss: {avg:.6f}")


# ============================================================
# EvaluateWorker
# 1. 输入的 pair 都是字符串
# 2. test 是一行一行去 test, 不是 batch
# 4. 支持 chrF 评分
# ============================================================
class EvaluateWorker:
    def __init__(self, model: nn.Module):
        self.model = model
        _device = get_device()
        self.model.to(_device)

    def test(
        self,
        test_pairs: list[tuple[str, str]],
        max_len_margin: int = 10,
    ) -> tuple[float, list[tuple[str, str, str, float]]]:
        """
        1. test_pairs 每个元素都是 (src_str, tgt_str), 纯字符串
        2. 逐行推理, 不做batch
        4. 对每条预测计算 chrF, 并返回平均值 + 详情

        返回: 
            avg_chrf: float               整个测试集的平均chrF
            results:  list[tuple[
                src_text: str,   # 源句（清洗后token）
                pred_text: str,  # 预测句
                ref_text: str,   # 参考句
                chrf: float,     # 该条chrF
            ]]
        """
        _model = self.model
        total_chrf = 0.0
        results: list[tuple[str, str, str, float]] = []

        with torch.no_grad():
            log_output_line(INFERENCE_START_TAG)
            pair_iter = tqdm(test_pairs, desc="Evaluating (row-by-row)")
            for src_str, ref_str in pair_iter:
                # ----- 字符串 → idx → 单条 tensor -----
                src_idx = tokenize_source(src_str)
                ref_idx = tokenize_target(ref_str)
                ref_with_eos = list(ref_idx) + [EOS_IDX]
                src_tensor = TensorHandler.src_idx_to_train_tensor(src_idx)

                # ----- 自回归逐行推理（不是 teacher forcing！）-----
                expected_len = len(ref_with_eos)
                max_len = max(expected_len + max_len_margin, 4)
                inf_logits = _model.inference(src_tensor, max_len=max_len)
                pred_idx = inf_logits.argmax(dim=-1).reshape(-1).tolist()

                # ----- idx → 文本 -----
                full_src_idx = [SOS_IDX] + list(src_idx) + [EOS_IDX]
                src_text = idx_list_to_token_source(full_src_idx)
                pred_text = idx_list_to_token_target(pred_idx)
                ref_text = idx_list_to_token_target(ref_with_eos)

                # ----- 计算 chrF -----
                score = chrf_score(pred_idx, ref_with_eos)
                total_chrf += score
                results.append((src_text, pred_text, ref_text, score))

                # ----- 写日志 -----
                log_output_line(
                    f"【chrF: {score:.4f}】\t[{src_text}]\t[{pred_text}]\t[{ref_text}]"
                )
            log_output_line(INFERENCE_END_TAG)

            avg_chrf = total_chrf / max(1, len(test_pairs))
            log_output_line(f"eval_avg_chrf: {avg_chrf:.6f}")

        return avg_chrf, results
