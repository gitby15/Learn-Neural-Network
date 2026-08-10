import re
import jieba


def fullwidth_to_halfwidth(text: str) -> str:
    """全角字符转半角（覆盖 ASCII 可打印字符区域）"""
    result = []
    for ch in text:
        code = ord(ch)
        # 全角 ASCII 范围：U+FF01 ~ U+FF5E（!~）
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        # 全角空格 U+3000 -> 半角空格
        elif code == 0x3000:
            result.append(' ')
        else:
            result.append(ch)
    return ''.join(result)

def split_eng_word(sentence: str) -> list[str]:
    return re.findall(r'[^\s.,!?;:"()\[\]{}\-&]+|[.,!?;:"()\[\]{}\-&]|\s', sentence)
def split_zh_word(sentence: str) -> list[str]:
    return list(jieba.cut(sentence,use_paddle=True))


if __name__ == "__main__":
    print(segment_zh_word("你好吗，我是lijin.tim，你好帅，你好像刘德华"))

