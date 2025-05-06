import re, string

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