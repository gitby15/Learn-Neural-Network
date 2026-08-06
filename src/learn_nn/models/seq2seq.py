import time
import random

import torch
from torch import nn
from tqdm import tqdm
from learn_nn.models._utils_ import log_output_line, save_model
from learn_nn.train_datasets.get_tatoeba import EOS_IDX, PAD_IDX, SOS_IDX, UNK_IDX, get_dataset, tokenize_source, tokenize_target, idx_list_to_token_source, idx_list_to_token_target

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("cuda available: ", torch.cuda.is_available(), torch.version.cuda)
print("mps available: ", torch.backends.mps.is_available())
print("using device:", DEVICE)

TRAIN_LIMIT = 5000
TEST_LIMIT = 50
BATCH_SIZE = 32
   

EMBED_SIZE = 256

class Encoder(nn.Module):
    def __init__(self, src_vocab):
        super().__init__()
        self.embedding = nn.Embedding(len(src_vocab), EMBED_SIZE)
        self.rnn = nn.GRU(EMBED_SIZE, EMBED_SIZE, batch_first=False)

    def forward(self, src):
        embedded = self.embedding(src)
        # 这里没有输出output，只输出隐藏状态
        # 因为解码器需要上一个时间步的隐藏状态作为输入
        _, hidden = self.rnn(embedded)
        return hidden


class ReserveEncoder(nn.Module):
    def __init__(self, src_vocab):
        super().__init__()
        self.embedding = nn.Embedding(len(src_vocab), EMBED_SIZE)
        # 双向 2 层 GRU：每个位置都能同时看过去和未来的上下文
        # 注：为了简化，保持 num_layers=1、bidirectional=True，后续加层再扩
        self.rnn = nn.GRU(
            EMBED_SIZE,
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


    
        
def get_tensor(sequences):
    tensor = torch.tensor(
        sequences,
        dtype=torch.long,
    ).unsqueeze(1)
    tensor = tensor.to(DEVICE)
    return tensor

def pad_sequences(sequences):
    max_len = max(len(sequence) for sequence in sequences)
    padded = []
    for sequence in sequences:
        token_ids = list(sequence)
        token_ids += [PAD_IDX] * (max_len - len(token_ids))
        padded.append(token_ids)
    tensor = torch.tensor(padded, dtype=torch.long, device=DEVICE)
    return tensor.transpose(0, 1)

def build_batch_tensors(batch_pairs, src_vocab, tgt_vocab, special_tokens):
    src_sequences = []
    tgt_input_sequences = []
    tgt_output_sequences = []
    for source_sequences, target_sequences in batch_pairs:
        src_tokens = [SOS_IDX] + tokenize_source(source_sequences) + [EOS_IDX]
        tgt_tokens = tokenize_target(target_sequences)
        src_sequences.append(src_tokens)
        tgt_input_sequences.append([SOS_IDX] + tgt_tokens)
        tgt_output_sequences.append(tgt_tokens + [EOS_IDX])
    src_tensor = pad_sequences(src_sequences)
    tgt_input_tensor = pad_sequences(tgt_input_sequences)
    tgt_output_tensor = pad_sequences(tgt_output_sequences)
    return src_tensor, tgt_input_tensor, tgt_output_tensor

def build_batches(pairs, batch_size, shuffle=True):
    ordered_pairs = list(pairs)
    if shuffle:
        random.shuffle(ordered_pairs)
    return [
        ordered_pairs[index:index + batch_size]
        for index in range(0, len(ordered_pairs), batch_size)
    ]

def prepare_batches(pairs, batch_size, src_vocab, tgt_vocab, special_tokens):
    pair_batches = build_batches(pairs, batch_size)
    prepared_batches = []
    batch_progress = tqdm(pair_batches, desc="Preparing batches")
    for batch_pairs in batch_progress:
        src_tensor, tgt_input_tensor, tgt_output_tensor = build_batch_tensors(
            batch_pairs,
            src_vocab,
            tgt_vocab,
            special_tokens
        )
        prepared_batches.append((src_tensor, tgt_input_tensor, tgt_output_tensor))
    return prepared_batches

def src_tensor_to_token(src_tensor, src_vocab, batch_index=0, skip_pad=True) -> str:
    inverse_vocab = {idx: token for token, idx in src_vocab.items()}

    if src_tensor.dim() == 1:
        token_ids = src_tensor.detach().cpu().tolist()
    elif src_tensor.dim() == 2:
        if batch_index >= src_tensor.size(1):
            raise IndexError(
                f"batch_index {batch_index} out of range for batch size {src_tensor.size(1)}"
            )
        token_ids = src_tensor[:, batch_index].detach().cpu().tolist()
    else:
        raise ValueError(f"expected 1D or 2D tensor, got shape {tuple(src_tensor.shape)}")

    tokens = []
    for token_id in token_ids:
        if skip_pad and token_id == PAD_IDX:
            continue
        tokens.append(inverse_vocab.get(token_id, "<UNK>"))
    return " ".join(tokens)

    

def align_logits_to_target(logits, target_tensor):
    if logits.size(0) < target_tensor.size(0):
        pad_len = target_tensor.size(0) - logits.size(0)
        pad_logits = logits.new_zeros(
            (pad_len, logits.size(1), logits.size(2)),
        )
        return torch.cat([logits, pad_logits], dim=0)
    if logits.size(0) > target_tensor.size(0):
        return logits[:target_tensor.size(0)]
    return logits


def main():
    pairs, test_pairs, src_vocab, tgt_vocab, special_tokens = get_dataset()

    model = Seq2SeqModel(src_vocab, tgt_vocab)
    model.to(DEVICE)

    train_batches = prepare_batches(pairs, BATCH_SIZE, src_vocab, tgt_vocab, special_tokens)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    epochs = 100
    progress = tqdm(range(epochs), desc="Training")

    log_output_line(f"=========== Train Meta: ===========")
    log_output_line(f"EMBED_SIZE: {EMBED_SIZE} | BATCH_SIZE: {BATCH_SIZE} | EPOCHS: {epochs}")
    log_output_line(f"TRAINING_PAIRS_LEN: {len(pairs)} | TEST_PAIRS_LEN: {len(test_pairs)}")
    log_output_line(f"=========== Train Meta End: ===========")
    for _ in progress:
        model.train()
        epoch_loss = 0.0
        epoch_batches = list(train_batches)
        random.shuffle(epoch_batches)
        for src_tensor, tgt_input_tensor, tgt_output_tensor in epoch_batches:

            logits = model(src_tensor, tgt_input_tensor)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output_tensor.reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        
        progress.set_postfix(loss=f"{epoch_loss / len(epoch_batches):.6f}")
        # print("")
        # print(f"epoch_loss: {epoch_loss / len(epoch_batches):.6f}")
    

    model.eval()
    save_model(model)
    eval_avg_loss = 0.0
    eval_count = 0
    with torch.no_grad():
        for index, test_pair in enumerate(test_pairs):
            source_sequences, target_sequences = test_pair

            src_seq = [SOS_IDX] + tokenize_source(source_sequences) + [EOS_IDX]
            tgt_tokens = tokenize_target(target_sequences)
            tgt_output_idx_list = tgt_tokens + [EOS_IDX]
            tgt_input_idx_list = [SOS_IDX] + tgt_tokens

            src_tensor = get_tensor(src_seq)
            tgt_input_tensor = get_tensor(tgt_input_idx_list)
            tgt_output_tensor = get_tensor(tgt_output_idx_list)

            eval_logits = model(src_tensor, tgt_input_tensor)
            eval_loss = criterion(
                eval_logits.reshape(-1, eval_logits.size(-1)),
                tgt_output_tensor.reshape(-1),
            )
            eval_avg_loss += eval_loss.item()
            eval_count += 1

            inf_logits = model.inference(
                src_tensor,
                max_len=len(tgt_output_idx_list) + 5,
            )
            inf_len = min(inf_logits.size(0), len(tgt_output_idx_list))
            inf_output = inf_logits[:inf_len].argmax(dim=-1).reshape(-1).tolist()
            output_tokens = idx_list_to_token_target(inf_output)

            expect_tokens = idx_list_to_token_target(tgt_output_idx_list)
            exact_match = "✅" if inf_output[:inf_len] == tgt_output_idx_list[:inf_len] else " "
            log_output_line(f"{exact_match}input: {idx_list_to_token_source(src_seq)} | tf_loss: {eval_loss.item():.6f}")
            log_output_line(f" === output: {output_tokens} | expect: {expect_tokens}==")

    log_output_line(f"eval_avg_loss (teacher-forcing): {eval_avg_loss / eval_count:.6f}")

if __name__ == "__main__":
    main()
