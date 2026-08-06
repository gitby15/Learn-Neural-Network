import re
import jieba


def split_eng_word(sentence: str) -> list[str]:
    return re.findall(r'[^\s.,!?;:"()\[\]{}\-&]+|[.,!?;:"()\[\]{}\-&]|\s', sentence)
def split_zh_word(sentence: str) -> list[str]:
    return list(jieba.cut(sentence,use_paddle=True))


if __name__ == "__main__":
    print(segment_zh_word("你好吗，我是lijin.tim，你好帅，你好像刘德华"))

