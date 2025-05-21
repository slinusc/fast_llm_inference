import re, string

_tok_re  = re.compile(r"\S+")
_sent_re = re.compile(r"[.!?…]+")

def tok_cnt(text: str) -> int:
    return len(_tok_re.findall(text))

def sent_cnt(text: str) -> int:
    return max(len(_sent_re.findall(text)), 1)

def chunker(seq, size):
            for i in range(0, len(seq), size):
                yield seq[i : i + size]

def normalize_answer(s: str) -> str:
            
            s = s.lower()
            s = re.sub(r'\b(a|an|the)\b', ' ', s)
            s = s.translate(str.maketrans('', '', string.punctuation))
            s = re.sub(r'\s+', ' ', s)
            return s.strip()

def clean_prediction(prediction: list[str]) -> list[str]:
    cleaned = []
    for raw in prediction:
        # 1) Remove anything after the first '###'
        ans = raw.split("###", 1)[0]
        # 2) Then remove anything after the first newline
        ans = ans.split("\n", 1)[0]
        # 3) Strip whitespace and trailing periods
        ans = ans.strip()
        cleaned.append(ans)
    return cleaned

