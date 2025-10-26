# rag/evaluator.py
import re
from collections import Counter

def normalize(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def nounish_tokens(text):
    toks = [w for w in normalize(text).split() if len(w) > 3]
    return Counter(toks)

def faithfulness(answer, evid_texts):
    # very rough: fraction of answer tokens supported by evidence tokens
    ans = nounish_tokens(answer)
    ev = Counter()
    for e in evid_texts:
        ev += nounish_tokens(e)
    if not ans: 
        return 1.0
    supported = sum(min(cnt, ev.get(tok,0)) for tok, cnt in ans.items())
    total = sum(ans.values())
    return supported / max(1, total)

def hallucination_rate(answers_with_evidence, thresh=0.6):
    bad = 0
    for ans, evid in answers_with_evidence:
        sc = faithfulness(ans, evid)
        if sc < thresh:
            bad += 1
    return bad / max(1, len(answers_with_evidence))