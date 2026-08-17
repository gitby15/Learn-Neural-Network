from learn_nn.models.mini_llm.dataset.minimind import PretrainDataset
from learn_nn.models.mini_llm.dataset.tokenizer.minimind_tokenizer import MinimindTokenizer
from learn_nn.models.mini_llm.model_code.model import MiniLLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import math
import random
import os


_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")
CHECKPOINT_FILE = "pretrain.pt"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)


BASE_LR = 3e-4
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.98)
EPS = 1e-9
WARMUP_RATIO = 0.1
GRAD_CLIP_NORM = 1.0


class PreTrainWorker:
    def __init__(self):
        self.tokenizer = MinimindTokenizer().get_tokenizer()
        self.dataset = PretrainDataset(tokenizer=self.tokenizer)
        _model = MiniLLM(len(self.tokenizer))
        self.model = torch.compile(_model)
        self.load_weight_times = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        use_fused = torch.cuda.is_available()
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=BASE_LR,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
            fused=use_fused,
        )

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer, total_steps: int
    ) -> LambdaLR:
        warmup_steps = int(total_steps * WARMUP_RATIO)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine_decay

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    @staticmethod
    def _compute_loss(logits, labels, criterion):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        scaler: torch.amp.GradScaler,
        epoch: int,
        global_step: int,
    ):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            },
            CHECKPOINT_PATH,
        )

    def _load_model_weights(self, model: nn.Module) -> bool:
        if not os.path.exists(CHECKPOINT_PATH):
            return False
        print(f"[Checkpoint] 从 {CHECKPOINT_PATH} 加载模型权重（继续预训练模式）...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Checkpoint] 模型权重加载成功（原训练进度: epoch={ckpt['epoch']}, step={ckpt['global_step']}）")
        self.load_weight_times += 1
        return True

    def _load_full_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        scaler: torch.amp.GradScaler,
    ) -> tuple[int, int]:
        if not os.path.exists(CHECKPOINT_PATH):
            return 0, 0
        print(f"[Checkpoint] 从 {CHECKPOINT_PATH} 完整恢复训练状态（断点续训模式）...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        print(f"[Checkpoint] 已恢复: epoch={start_epoch}, global_step={global_step}")
        return start_epoch, global_step

    def train(self, pairs_batches: list, resume: bool = True, epochs: int=100):
        """
        Args:
            pairs_batches: 训练数据批次列表
            resume: True=断点续训（恢复所有状态，从上次中断的epoch继续）
                    False=继续预训练（只加载模型权重，optimizer/scheduler/epoch全部重置，
                           适用于用更多数据或新数据继续训练）
        """
        model = self.model
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        optimizer = self._build_optimizer()
        total_steps = epochs * len(pairs_batches)
        scheduler = self._build_scheduler(optimizer, total_steps)
        scaler = torch.amp.GradScaler("cuda")

        if resume:
            start_epoch, global_step = self._load_full_state(
                model, optimizer, scheduler, scaler
            )
        else:
            self._load_model_weights(model)
            start_epoch, global_step = 0, 0

        model.train()
        progress = tqdm(range(start_epoch, epochs), desc="Training")

        for epoch in progress:
            epoch_loss = 0.0
            random.shuffle(pairs_batches)

            for input_ids, labels in pairs_batches:
                with torch.amp.autocast("cuda"):
                    logits = model(input_ids)
                    loss = self._compute_loss(logits, labels, criterion)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                global_step += 1
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(pairs_batches))
            cur_lr = optimizer.param_groups[0]["lr"]
            progress.set_postfix(loss=f"{avg_loss:.6f}", lr=f"{cur_lr:.2e}")

            self._save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, global_step)

    @torch.no_grad()
    def generate(
        self,
        input_str_batch: list[str],
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> list[str]:
        model = self.model
        tokenizer = self.tokenizer
        device = next(model.parameters()).device
        model.eval()

        results = []
        for text in input_str_batch:
            input_ids = [tokenizer.bos_token_id] + tokenizer(
                text, add_special_tokens=False
            )["input_ids"]
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

            for _ in range(max_new_tokens):
                logits = model(input_ids)
                next_logits = logits[:, -1, :]

                if temperature > 0 and temperature != 1.0:
                    next_logits = next_logits / temperature

                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float("-inf")

                if temperature == 0:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

                input_ids = torch.cat([input_ids, next_token], dim=1)

            gen_ids = input_ids[0].tolist()
            prompt_token_count = 1 + len(tokenizer(text, add_special_tokens=False)["input_ids"])
            gen_ids = gen_ids[prompt_token_count:]
            gen_ids = [t for t in gen_ids if t not in (tokenizer.pad_token_id, tokenizer.eos_token_id)]
            results.append(tokenizer.decode(gen_ids))
        return results

    def inference(self, input_str_batch: list[str]) -> list[str]:
        if os.path.exists(CHECKPOINT_PATH) and self.load_weight_times == 0:
            print("从磁盘读取模型权重：", CHECKPOINT_PATH)
            self._load_model_weights(self.model)
        return self.generate(input_str_batch)

    def evaluate(self, pairs: list):
        self.model.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for input_idx_batch, label_batch in pairs:
                logits = self.model(input_idx_batch)
                shift_labels = label_batch[..., 1:].contiguous()
                n_tokens = shift_labels.ne(-100).sum().item()
                shift_logits = logits[..., :-1, :].contiguous()
                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                total_loss += loss.item()
                total_tokens += n_tokens
        avg_loss = total_loss / max(1, total_tokens)
        return avg_loss
