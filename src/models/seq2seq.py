import random

import torch
from torch import nn
from tqdm import tqdm

from train_datasets.get_tatoeba import EOS_IDX, PAD_IDX, SOS_IDX, UNK_IDX, get_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("using device:", DEVICE)

TRAIN_LIMIT = 5000
TEST_LIMIT = 50
BATCH_SIZE = 32
   
HIDDEN_SIZE = 64
EMBED_SIZE = 32

  
   

def index_to_token(index, vocab):
    return next(token for token, idx in vocab.items() if idx == index)


def tokenize_source(text):
    return text.strip().split()


def tokenize_target(text):
    return list(text.strip())


class Encoder(nn.Module):
    def __init__(self, src_vocab):
        super().__init__()
        self.embedding = nn.Embedding(len(src_vocab), EMBED_SIZE)
        self.rnn = nn.GRU(EMBED_SIZE, HIDDEN_SIZE, batch_first=False)

    def forward(self, src):
        embedded = self.embedding(src)
        # 这里没有输出output，只输出隐藏状态
        # 因为解码器需要上一个时间步的隐藏状态作为输入
        _, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, tgt_vocab):
        super().__init__()
        self.embedding = nn.Embedding(len(tgt_vocab), EMBED_SIZE)
        self.rnn = nn.GRU(EMBED_SIZE, HIDDEN_SIZE, batch_first=False)
        self.linear = nn.Linear(HIDDEN_SIZE, len(tgt_vocab))
    
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
        self.encoder = Encoder(src_vocab)
        self.decoder = Decoder(tgt_vocab)

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


    
        
def get_tensor(sequences, vocab):
    tensor = torch.tensor(
        [vocab.get(token, UNK_IDX) for token in sequences],
        dtype=torch.long,
    ).unsqueeze(1)
    tensor = tensor.to(DEVICE)
    return tensor

def pad_sequences(sequences, vocab):
    max_len = max(len(sequence) for sequence in sequences)
    padded = []
    for sequence in sequences:
        token_ids = [vocab.get(token, UNK_IDX) for token in sequence]
        token_ids += [PAD_IDX] * (max_len - len(token_ids))
        padded.append(token_ids)
    tensor = torch.tensor(padded, dtype=torch.long, device=DEVICE)
    return tensor.transpose(0, 1)

def build_batch_tensors(batch_pairs, src_vocab, tgt_vocab, special_tokens):
    src_sequences = []
    tgt_input_sequences = []
    tgt_output_sequences = []
    for source_sequences, target_sequences in batch_pairs:
        src_tokens = [special_tokens[1]] + tokenize_source(source_sequences) + [special_tokens[2]]
        tgt_tokens = tokenize_target(target_sequences)
        src_sequences.append(src_tokens)
        tgt_input_sequences.append([special_tokens[1]] + tgt_tokens)
        tgt_output_sequences.append(tgt_tokens + [special_tokens[2]])
    src_tensor = pad_sequences(src_sequences, src_vocab)
    tgt_input_tensor = pad_sequences(tgt_input_sequences, tgt_vocab)
    tgt_output_tensor = pad_sequences(tgt_output_sequences, tgt_vocab)
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

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 30
    progress = tqdm(range(epochs), desc="Training")


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


    model.eval()
    inference_avg_loss = 0.0
    inference_count = 0
    with torch.no_grad():
        test_progress = tqdm(test_pairs, desc="Testing")
        for test_pair in test_progress:
            source_sequences, target_sequences = test_pair
            
            src_seq = [special_tokens[1]] + tokenize_source(source_sequences) + [special_tokens[2]]
            tgt_output_seq = tokenize_target(target_sequences) + [special_tokens[2]]

            src_tensor = get_tensor(src_seq, src_vocab)
            tgt_output_tensor = get_tensor(tgt_output_seq, tgt_vocab)
            logits = model.inference(
                src_tensor,
                max_len=max(len(src_seq) + 5, len(tgt_output_seq)),
            )
            logits = align_logits_to_target(logits, tgt_output_tensor)
            output = logits.argmax(dim=-1).reshape(-1).tolist()
            output_tokens = [index_to_token(index, tgt_vocab) for index in output]
            inference_loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output_tensor.reshape(-1),
            )
            inference_avg_loss += inference_loss.item()
            inference_count += 1
            progress.set_postfix(loss=f"{inference_loss.item():.6f}")
        
    print(f"inference_avg_loss: {inference_avg_loss/inference_count:.6f}")

if __name__ == "__main__":
    main()
