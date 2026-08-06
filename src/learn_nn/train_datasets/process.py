from learn_nn.train_datasets._utils_ import split_eng_word, split_zh_word
from pathlib import Path





FOLDER_PATH = Path(__file__).resolve().parent
FILE_PATH = FOLDER_PATH / "tatoeba_en2zh_tgt.tsv"
SORTED_FILE_PATH = FOLDER_PATH / "tatoeba_en2zh_sorted.tsv"
SIMPLE_SORTED_FILE_PATH = FOLDER_PATH / "tatoeba_en2zh_sorted_simple.tsv"

def read_file(file_path: Path) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines

def write_file(file_path: Path, lines: list[str]):
    with open(file_path, "w" , encoding="utf-8") as f:
        f.writelines(lines)


def sort_by_english_word_count():
    lines = read_file(FILE_PATH)

    def _english_word_count(line: str) -> int:
        parts = line.split('\t')
        english_sentence = parts[0]
        return len(split_eng_word(english_sentence))

    lines.sort(key=_english_word_count)
    _lines = []
    for line in lines:
        line = line[:-1]
        line = line + '\t\n'
        _lines.append(line)

    
    # write(TGT_FILE_PATH, _lines)
    pass

def convert_to_simple():
    from zhconv import convert
    lines = read_file(SORTED_FILE_PATH)
    _lines = []
    for line in lines:
        line = line.split('\t')
        simple_zh_sentence = convert(line[1], 'zh-cn')
        line[1] = simple_zh_sentence
        line = '\t'.join(line)
        _lines.append(line)
    write_file(SIMPLE_SORTED_FILE_PATH, _lines)

def main():
    convert_to_simple()

if __name__ == "__main__":
    main()