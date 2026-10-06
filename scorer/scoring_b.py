"""역할 B(문장 만들기) 채점과 코퍼스."""
from pathlib import Path

from . import checks
from .lm import LSI, BigramLM, ref_coverage
from .metrics import distribution


def build_corpus(data):
    """입력 대화의 챗봇 문장(중복 제거, 처음 나온 순서)에서 참조 문장을 뺀 것.

    챗봇 문장은 response.jsonl의 대화에서 읽고, 참조 문장은 그 completion이다.
    문자열이 정확히 같은 줄만 뺀다.
    """
    references = set(data.references())
    seen = set()
    sentences = []
    for case_id in data.ids:
        for message in data.user_json(data.row_b(case_id))["context"]:
            if message.get("role") != "assistant":
                continue
            text = message["content"]
            if text in seen or text in references:
                seen.add(text)
                continue
            seen.add(text)
            sentences.append(text)
    for s in sentences:
        if "\n" in s or "\r" in s:
            raise ValueError("a corpus sentence contains a line break; one sentence per line would break")
    return sentences


def write_corpus(sentences, path):
    """코퍼스 문장을 한 줄에 하나씩 파일로 쓴다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(s + "\n" for s in sentences), encoding="utf-8")


def read_corpus(path):
    """코퍼스 파일을 읽어 빈 줄을 뺀 문장 목록으로 돌려준다."""
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    return [line for line in lines if line.strip()]


class SentenceScorer:
    """코퍼스로 언어모델(FM용)과 LSI(AM용)를 만들어 두고 후보 문장을 채점한다. (ngram 백엔드)"""
    kind = "ngram"
    info = {"type": "ngram"}

    def __init__(self, corpus_sentences, lsi_dim=10):
        """코퍼스 문장으로 bigram 언어모델과 LSI 공간을 만든다."""
        self.corpus_size = len(corpus_sentences)
        self.lm = BigramLM(corpus_sentences)
        self.lsi = LSI(corpus_sentences, dim=lsi_dim)

    def score(self, candidate, references):
        """후보의 FM, AM, oov_word, oov_bigram. 참조가 여럿이면 가장 높은 값(최댓값)을 쓴다."""
        if not isinstance(candidate, str):
            return {"FM": 0.0, "AM": 0.0, "oov_word": None, "oov_bigram": None}
        if not candidate.split():
            return {"FM": 0.0, "AM": 0.0, "oov_word": None, "oov_bigram": None}
        fm = max(self.lm.fm(candidate, r) for r in references)
        am = max(self.lsi.am(candidate, r) for r in references)
        oov_word, oov_bigram = self.lm.oov_rates(candidate)
        return {"FM": fm, "AM": am, "oov_word": oov_word, "oov_bigram": oov_bigram}


SCORE_KEYS = ("id", "missing", "FM", "AM", "oov_word", "oov_bigram", "completion_claim")


def score_run(data, outputs, scorer):
    """모든 사례를 채점한다. outputs는 id에서 후보로의 dict이고, 없는 id는 결측이다.

    completion_claim(계획 위반 표현)은 보조 지표라서 FM·AM에 영향을 주지 않는다.
    """
    rows = []
    for case_id in data.ids:
        missing = case_id not in outputs or not isinstance(outputs[case_id], str)
        candidate = None if missing else outputs[case_id]
        scores = scorer.score(candidate, [data.reference(case_id)])
        claim = checks.has_completion_claim(candidate, data.plan_action(case_id))
        row = {"id": case_id, "missing": missing, **scores, "completion_claim": claim}
        assert tuple(row) == SCORE_KEYS
        rows.append(row)
    return rows


def _mean(values):
    """None을 뺀 평균. 값이 하나도 없으면 None."""
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(rows, data, scorer, min_sentences):
    """B 실행 하나의 요약(m2p_summary.json). FM·AM 평균과 분포, 결측 수, 사용한 문장 점수 방식(scorer)을 담는다.

    ngram 방식이면 미등록 비율 평균, 코퍼스 규모와 '미달' 여부, ref_coverage도 담는다. neural 방식은 코퍼스를 쓰지
    않으므로 이 값들이 None이고 corpus_below_minimum은 False이다.
    """
    refs = data.references()
    n = len(rows)
    ngram = scorer.kind == "ngram"
    return {
        "n_cases": n,
        "scorer": scorer.info,
        "fm_mean": _mean([r["FM"] for r in rows]),
        "am_mean": _mean([r["AM"] for r in rows]),
        "fm_distribution": distribution([r["FM"] for r in rows]),
        "am_distribution": distribution([r["AM"] for r in rows]),
        "oov_word_mean": _mean([r["oov_word"] for r in rows]),
        "oov_bigram_mean": _mean([r["oov_bigram"] for r in rows]),
        "missing_outputs": sum(1 for r in rows if r["missing"]),
        "completion_claims": sum(1 for r in rows if r["completion_claim"]),
        "completion_claim_rate": (sum(1 for r in rows if r["completion_claim"])
                                  / sum(1 for r in rows if not r["missing"])
                                  if any(not r["missing"] for r in rows) else None),
        "corpus_sentences": scorer.corpus_size if ngram else None,
        "corpus_below_minimum": scorer.corpus_size < min_sentences if ngram else False,
        "corpus_minimum": min_sentences if ngram else None,
        "ref_coverage": ref_coverage(scorer.lm, refs) if ngram else None,
        "reference_count_per_case": 1,
        "distinct_references": len(set(refs)),
    }
