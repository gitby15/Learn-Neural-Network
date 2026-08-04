from fcntl import F_GETPATH
import torch
from pathlib import Path

FOLDER_PATH = Path(__file__).resolve().parent
FILE_PATH = FOLDER_PATH / "tatoeba_en2zh_tgt.tsv"

PAD_TOKEN = "<PAD>"
PAD_IDX = 0

SOS_TOKEN = "<SOS>"
SOS_IDX = 1

EOS_TOKEN = "<EOS>"
EOS_IDX = 2

UNK_TOKEN = "<UNK>"
UNK_IDX = 3





# (src_sentence, tgt_sentence)
_tarin_pairs = []
_test_pairs = []

_src_vocab = {PAD_TOKEN: PAD_IDX, SOS_TOKEN: SOS_IDX, EOS_TOKEN: EOS_IDX, UNK_TOKEN: UNK_IDX}
_tgt_vocab = {PAD_TOKEN: PAD_IDX, SOS_TOKEN: SOS_IDX, EOS_TOKEN: EOS_IDX, UNK_TOKEN: UNK_IDX}



def read_file(file_path: Path) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines

def build_vocab(pairs: list[tuple[str, str]]):
    for src, tgt in pairs:
        for src_token in src.split(' '):
            if src_token not in _src_vocab:
                _src_vocab[src_token] = len(_src_vocab)
        for tgt_token in tgt.strip():
            if tgt_token not in _tgt_vocab:
                _tgt_vocab[tgt_token] = len(_tgt_vocab)
    

def build_pairs():
    lines = read_file(FILE_PATH)
    _split_idx = 50
    for line in lines[:-_split_idx]:
        temp = line.split("\t")
        _tarin_pairs.append((temp[0], temp[1]))

    for line in lines[-_split_idx:]:
        temp = line.split("\t")
        _test_pairs.append((temp[0], temp[1]))
    build_vocab(_tarin_pairs + _test_pairs)


def get_dataset():

    build_pairs()

    _special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    return _tarin_pairs, _test_pairs, _src_vocab, _tgt_vocab, _special_tokens


def test():
    build_pairs()
    tgt = _tgt_vocab
    print(tgt)

if __name__ == "__main__":
    test()
