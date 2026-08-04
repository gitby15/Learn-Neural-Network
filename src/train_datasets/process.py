import re
from pathlib import Path

FOLDER_PATH = Path(__file__).resolve().parent
FILE_PATH = FOLDER_PATH / "tatoeba_en2zh_tgt.tsv"
TGT_FILE_PATH = FOLDER_PATH / "tatoeba_en2zh_sorted.tsv"

def read_file(file_path: Path) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines

def write(file_path: Path, lines: list[str]):
    with open(file_path, "w" , encoding="utf-8") as f:
        f.writelines(lines)

def split_eng_word(sentence: str) -> list[str]:
    sentence = re.sub(r'([.,!?;:"()\[\]{}\-&])', r' \1 ', sentence)
    return [token for token in sentence.split(' ') if token]

def main():
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

    
    write(TGT_FILE_PATH, _lines)
    pass


if __name__ == "__main__":
    main()