# 暂时先直接用minimnd的Tokenizer，先跑通流程，再回来啃这一块

import os
from transformers import AutoTokenizer

folder_path = os.path.dirname(__file__)

class MinimindTokenizer():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(folder_path)
    def get_tokenizer(self):
        return self.tokenizer
    def __call__(self, text: str):
        return self.tokenizer(text)
    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id
    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size


if __name__ == "__main__":
    tokenizer = MinimindTokenizer()
    print(tokenizer.get_pad_id())
    print(tokenizer.get_vocab_size())