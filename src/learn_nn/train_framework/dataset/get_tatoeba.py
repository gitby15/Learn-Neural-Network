from pathlib import Path

from learn_nn.train_framework.dataset._utils_ import split_eng_word, split_zh_word

FOLDER_PATH = Path(__file__).resolve().parent
FILE_PATH = FOLDER_PATH / "tatoeba_en2zh_sorted_simple.tsv"

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
_src_vocab_idx = {PAD_IDX: PAD_TOKEN, SOS_IDX: SOS_TOKEN, EOS_IDX: EOS_TOKEN, UNK_IDX: UNK_TOKEN}

_tgt_vocab = {PAD_TOKEN: PAD_IDX, SOS_TOKEN: SOS_IDX, EOS_TOKEN: EOS_IDX, UNK_TOKEN: UNK_IDX}
_tgt_vocab_idx = {PAD_IDX: PAD_TOKEN, SOS_IDX: SOS_TOKEN, EOS_IDX: EOS_TOKEN, UNK_IDX: UNK_TOKEN}



# 英文字符串转token列表
def tokenize_source(sentence: str)-> list[int]:
    return [_src_vocab.get(token, UNK_IDX) for token in split_eng_word(sentence)]

def idx_list_to_token_source(idx_list: list[int]) -> str:
    return "".join([_src_vocab_idx.get(idx, UNK_TOKEN) for idx in idx_list])

def tokenize_target(sentence: str)-> list[int]:
    return [_tgt_vocab.get(token, UNK_IDX) for token in split_zh_word(sentence)]

def idx_list_to_token_target(idx_list: list[int]) -> str:
    return "".join([_tgt_vocab_idx.get(idx, UNK_TOKEN) for idx in idx_list])


def read_file(file_path: Path) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines

def build_vocab(pairs: list[tuple[str, str]]):
    print(f"origin pairs len: {len(pairs)}")
    for src, tgt in pairs:
        for src_token in split_eng_word(src):
            if src_token not in _src_vocab:
                _idx = len(_src_vocab)
                _src_vocab[src_token] = _idx
                _src_vocab_idx[_idx] = src_token
        for tgt_token in split_zh_word(tgt):
            if tgt_token not in _tgt_vocab:
                _idx = len(_tgt_vocab)
                _tgt_vocab[tgt_token] = _idx
                _tgt_vocab_idx[_idx] = tgt_token

    print(f"build vocab done, src_vocab len: {len(_src_vocab)}, tgt_vocab len: {len(_tgt_vocab)}")
    print(f"build vocab idx done, src_vocab_idx len: {len(_src_vocab_idx)}, tgt_vocab_idx len: {len(_tgt_vocab_idx)}")

def build_pairs(min_len, max_len):
    lines = read_file(FILE_PATH)
    _filter_len_min = min_len
    _filter_len_max = max_len
    test_pair_ratio = 10

    def _english_word_count(line: str) -> int:
        parts = split_eng_word(line)
        return len(parts)

    
    for _index, line in enumerate(lines):
        line = line.split('\t')
        _count = _english_word_count(line[0])
        if _filter_len_min > _count or _filter_len_max < _count:
            continue
        if _index%test_pair_ratio == 0:
            _test_pairs.append((line[0].lower(), line[1]))
        else:
            _tarin_pairs.append((line[0].lower(), line[1]))
        _index += 1
        
        
    print(f"origin_len: {len(lines)}, train_pairs_len: {len(_tarin_pairs)}, test_pairs_len: {len(_test_pairs)}")
    build_vocab(_tarin_pairs + _test_pairs)


def get_dataset(min_len=0, max_len=9999):
    build_pairs(min_len, max_len)
    return _tarin_pairs, _test_pairs, _src_vocab, _tgt_vocab


def test():
    pairs, test_pairs, src_vocab, tgt_vocab = get_dataset()
    for pair in pairs:
            # src_sentence = pair[0]
            # idx_list = tokenize_source(src_sentence)
            # print("s -> i: ", src_sentence, ' -> ', idx_list)
            # print("i -> s: ", idx_list, ' -> ', idx_list_to_token_source(idx_list))

            tgt_sentence = pair[1]
            idx_list_tgt = tokenize_target(tgt_sentence)
            print('====')
            print("t -> i: ", tgt_sentence, ' -> ', idx_list_tgt)
            print("i -> t: ", idx_list_tgt, ' -> ', idx_list_to_token_target(idx_list_tgt))
            

if __name__ == "__main__":
    test()
