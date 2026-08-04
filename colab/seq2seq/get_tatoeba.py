from __future__ import annotations

import csv
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<padding>"
SOS_TOKEN = "<start_of_sequence>"
EOS_TOKEN = "<end_of_sequence>"
UNK_TOKEN = "<unknown>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

DEFAULT_FILENAME = "tatoeba-en2zh.tsv"
DEFAULT_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/gitby15/Learn-Neural-Network/"
    "main/colab/seq2seq/tatoeba-en2zh.tsv"
)


def tokenize_source(text: str) -> list[str]:
    return text.strip().split()


def tokenize_target(text: str) -> list[str]:
    return [char for char in text.replace(" ", "").strip()]


def default_data_path() -> Path:
    return Path(__file__).resolve().parent / DEFAULT_FILENAME


def ensure_tatoeba_file(
    file_path: str | Path | None = None,
    download_url: str = DEFAULT_DOWNLOAD_URL,
    download_if_missing: bool = True,
) -> Path:
    path = Path(file_path) if file_path is not None else default_data_path()
    if path.exists():
        return path

    if not download_if_missing:
        raise FileNotFoundError(f"Tatoeba data file not found: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(download_url, path)
    return path


@dataclass(frozen=True)
class TatoebaExample:
    source_text: str
    target_text: str
    source_tokens: list[str]
    target_tokens: list[str]
    source_id: str | None = None
    target_id: str | None = None


class TatoebaTranslationDataset(Dataset[TatoebaExample]):
    def __init__(self, examples: list[TatoebaExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TatoebaExample:
        return self.examples[index]

    def to_pairs(self) -> list[tuple[list[str], list[str]]]:
        return [
            (example.source_tokens, example.target_tokens)
            for example in self.examples
        ]


def read_tatoeba_examples(
    file_path: str | Path | None = None,
    limit: int | None = None,
    source_tokenizer: Callable[[str], list[str]] = tokenize_source,
    target_tokenizer: Callable[[str], list[str]] = tokenize_target,
    deduplicate: bool = True,
    download_if_missing: bool = True,
    download_url: str = DEFAULT_DOWNLOAD_URL,
) -> list[TatoebaExample]:
    path = ensure_tatoeba_file(
        file_path=file_path,
        download_url=download_url,
        download_if_missing=download_if_missing,
    )

    examples: list[TatoebaExample] = []
    seen_pairs: set[tuple[str, str]] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.reader(fp, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue

            source_id, source_text, target_id, target_text = row[:4]
            source_text = source_text.strip()
            target_text = target_text.strip()
            if not source_text or not target_text:
                continue

            pair_key = (source_text, target_text)
            if deduplicate and pair_key in seen_pairs:
                continue

            source_tokens = source_tokenizer(source_text)
            target_tokens = target_tokenizer(target_text)
            if not source_tokens or not target_tokens:
                continue

            seen_pairs.add(pair_key)
            examples.append(
                TatoebaExample(
                    source_id=source_id or None,
                    target_id=target_id or None,
                    source_text=source_text,
                    target_text=target_text,
                    source_tokens=source_tokens,
                    target_tokens=target_tokens,
                )
            )
            if limit is not None and len(examples) >= limit:
                break

    return examples


def split_examples(
    examples: list[TatoebaExample],
    train_ratio: float = 0.98,
    valid_ratio: float = 0.01,
    test_ratio: float = 0.01,
    seed: int = 42,
) -> tuple[list[TatoebaExample], list[TatoebaExample], list[TatoebaExample]]:
    total_ratio = train_ratio + valid_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1.0")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    total_count = len(shuffled)
    train_end = int(total_count * train_ratio)
    valid_end = train_end + int(total_count * valid_ratio)

    train_examples = shuffled[:train_end]
    valid_examples = shuffled[train_end:valid_end]
    test_examples = shuffled[valid_end:]
    return train_examples, valid_examples, test_examples


def build_vocab(examples: list[TatoebaExample]) -> tuple[dict[str, int], dict[str, int]]:
    src_vocab = {
        PAD_TOKEN: PAD_IDX,
        SOS_TOKEN: SOS_IDX,
        EOS_TOKEN: EOS_IDX,
        UNK_TOKEN: UNK_IDX,
    }
    tgt_vocab = {
        PAD_TOKEN: PAD_IDX,
        SOS_TOKEN: SOS_IDX,
        EOS_TOKEN: EOS_IDX,
        UNK_TOKEN: UNK_IDX,
    }

    for example in examples:
        for token in example.source_tokens:
            if token not in src_vocab:
                src_vocab[token] = len(src_vocab)
        for token in example.target_tokens:
            if token not in tgt_vocab:
                tgt_vocab[token] = len(tgt_vocab)

    return src_vocab, tgt_vocab


def tokens_to_ids(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    return [vocab.get(token, UNK_IDX) for token in tokens]


def pad_token_ids(
    sequences: list[list[str]],
    vocab: dict[str, int],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    max_len = max(len(sequence) for sequence in sequences)
    padded_token_ids = []
    for sequence in sequences:
        token_ids = tokens_to_ids(sequence, vocab)
        token_ids += [PAD_IDX] * (max_len - len(token_ids))
        padded_token_ids.append(token_ids)
    return torch.tensor(padded_token_ids, dtype=torch.long, device=device).transpose(0, 1)


class Seq2SeqCollator:
    def __init__(
        self,
        src_vocab: dict[str, int],
        tgt_vocab: dict[str, int],
        device: torch.device | str | None = None,
    ):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device

    def __call__(
        self,
        batch: list[TatoebaExample],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_sequences = []
        tgt_input_sequences = []
        tgt_output_sequences = []
        for example in batch:
            src_sequences.append([SOS_TOKEN] + example.source_tokens + [EOS_TOKEN])
            tgt_input_sequences.append([SOS_TOKEN] + example.target_tokens)
            tgt_output_sequences.append(example.target_tokens + [EOS_TOKEN])

        src_tensor = pad_token_ids(src_sequences, self.src_vocab, device=self.device)
        tgt_input_tensor = pad_token_ids(
            tgt_input_sequences,
            self.tgt_vocab,
            device=self.device,
        )
        tgt_output_tensor = pad_token_ids(
            tgt_output_sequences,
            self.tgt_vocab,
            device=self.device,
        )
        return src_tensor, tgt_input_tensor, tgt_output_tensor


def build_tatoeba_datasets(
    file_path: str | Path | None = None,
    limit: int | None = None,
    train_ratio: float = 0.98,
    valid_ratio: float = 0.01,
    test_ratio: float = 0.01,
    seed: int = 42,
    deduplicate: bool = True,
    download_if_missing: bool = True,
    download_url: str = DEFAULT_DOWNLOAD_URL,
    source_tokenizer: Callable[[str], list[str]] = tokenize_source,
    target_tokenizer: Callable[[str], list[str]] = tokenize_target,
) -> tuple[
    TatoebaTranslationDataset,
    TatoebaTranslationDataset,
    TatoebaTranslationDataset,
    dict[str, int],
    dict[str, int],
]:
    examples = read_tatoeba_examples(
        file_path=file_path,
        limit=limit,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        deduplicate=deduplicate,
        download_if_missing=download_if_missing,
        download_url=download_url,
    )
    train_examples, valid_examples, test_examples = split_examples(
        examples,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    src_vocab, tgt_vocab = build_vocab(train_examples + valid_examples + test_examples)
    return (
        TatoebaTranslationDataset(train_examples),
        TatoebaTranslationDataset(valid_examples),
        TatoebaTranslationDataset(test_examples),
        src_vocab,
        tgt_vocab,
    )


def preview_dataset(
    dataset: TatoebaTranslationDataset,
    count: int = 3,
) -> None:
    for index, example in enumerate(dataset.examples[:count]):
        print(f"[{index}] {example.source_text} -> {example.target_text}")


def main() -> None:
    train_ds, valid_ds, test_ds, src_vocab, tgt_vocab = build_tatoeba_datasets()
    print(f"train size: {len(train_ds)}")
    print(f"valid size: {len(valid_ds)}")
    print(f"test size: {len(test_ds)}")
    print(f"source vocab size: {len(src_vocab)}")
    print(f"target vocab size: {len(tgt_vocab)}")
    preview_dataset(train_ds)


if __name__ == "__main__":
    main()
