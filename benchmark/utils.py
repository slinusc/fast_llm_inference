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

        # 2) Strip whitespace (including newlines) from both ends
        ans = ans.strip()

        # 3) Remove anything after the first newline (in the stripped string)
        ans = ans.split("\n", 1)[0]

        # 4) Strip again and remove any trailing periods
        ans = ans.strip().rstrip(".")

        cleaned.append(ans)
    return cleaned

