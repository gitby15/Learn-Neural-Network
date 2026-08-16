
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import math

class PreTrainWorker():
    def __init__(self, model:nn.Module, epochs: int = 10):
        self.epochs = epochs
        self.model = model

    # pairs需要输入的形状是([B,T,C], [B,T,C])
    def train(self, pairs: list):
        _model = self.model
        criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)
        # ---- Transformer 推荐的优化器与学习率调度 ----
        # 1) 用 AdamW (带 decoupled weight decay) 替代 Adam
        BASE_LR = 3e-4       # 比 0.001 稍稳一些，可调 1e-4 ~ 5e-4
        WEIGHT_DECAY = 0.01  # Transformer 标配
        WARMUP_RATIO = 0.1   # 前 10% steps 做 warmup        
        optimizer = torch.optim.AdamW(
            _model.parameters(),
            lr=BASE_LR,
            betas=(0.9, 0.98),   # 原始 Transformer 论文推荐
            eps=1e-9,
            weight_decay=WEIGHT_DECAY,
        )

        # 2) 总训练步数 = epoch 数 * 每 epoch 的 batch 数
        total_steps = self.epochs * max(1, 1)
        warmup_steps = int(total_steps * WARMUP_RATIO)

        # 3) 学习率曲线: warmup 线性上升 → cosine 衰减到 0
        def _lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # warmup: 线性从 0 -> 1 (再乘 BASE_LR 就是实际 lr)
                return float(current_step) / float(max(1, warmup_steps))
            # cosine 从 1 衰减到 0
            progress_step = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(1e-4, 0.5 * (1.0 + math.cos(math.pi * progress_step)))

        scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)

        progress = tqdm(range(self.epochs), desc="Training")
        
        global_step = 0
        _model.train()
        for idx in progress:
            epoch_loss = 0.0
            # random.shuffle(epoch_batches)
            for input_idx_batch, label_batch in pairs:
                # pretrain.py 的 loss 计算处
                logits = _model(input_idx_batch)           # [B, T, V]
                loss = criterion_ce(
                    logits.view(-1, logits.size(-1)),
                    label_batch.view(-1),
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(_model.parameters(), max_norm=1.0)
                optimizer.step()
                # 每个 batch step 更新一次学习率 (step-based 而非 epoch-based)
                scheduler.step()
                global_step += 1
                epoch_loss += loss.item()
            avg = epoch_loss / max(1, len(pairs))
            cur_lr = optimizer.param_groups[0]["lr"]
            progress.set_postfix(loss=f"{avg:.6f}", lr=f"{cur_lr:.2e}")        


