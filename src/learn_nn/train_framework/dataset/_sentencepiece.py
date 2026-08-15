import sentencepiece as spm

from pathlib import Path

current_dir = Path(__file__).resolve().parent


def main():
    input_path = current_dir / "tatoeba_en2zh_sorted_simple.tsv"
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix="m"
    )



if __name__ == "__main__":
    main()