"""역할 B의 문장 품질 모델: bigram 언어모델(FM)과 LSI 공간(AM).

모든 값은 넘겨받은 코퍼스에서 계산한다. 코퍼스 크기에 의존하는 상수는 없다.
"""
import math
from collections import Counter, defaultdict

import numpy as np

START = "<s>"
TINY = 1e-12


def tokenize(text):
    """공백 문자(유니코드 공백 포함)로 나눈 어절. 정규화나 구두점 제거는 하지 않는다."""
    return text.split()


class BigramLM:
    """Good-Turing(Katz) bigram 모델. 후퇴는 add-one unigram으로 한다."""

    def __init__(self, sentences):
        """코퍼스 문장에서 bigram과 unigram 횟수를 세고 Good-Turing 할인 계수 d_1~d_5를 계산한다. 어절이 하나도 없으면
        ValueError.
        """
        self.bigrams = Counter()
        self.unigrams = Counter()
        for sentence in sentences:
            tokens = tokenize(sentence)
            if not tokens:
                continue
            padded = [START] + tokens
            self.bigrams.update(zip(padded, padded[1:]))     # Counter.update는 항목이 나온 횟수를 더한다
            self.unigrams.update(tokens)
        if not self.unigrams:
            raise ValueError("the corpus has no tokens")
        self.vocab = set(self.unigrams)
        self.total = sum(self.unigrams.values())          # T: <s>를 뺀 전체 어절 수
        self.v = len(self.vocab)                            # V: <s>를 뺀 어휘 수
        self.context_total = Counter()
        self.following = defaultdict(list)
        for (a, b), count in self.bigrams.items():
            self.context_total[a] += count
            self.following[a].append((b, count))
        n_r = Counter(self.bigrams.values())
        self.discount = {}
        for r in range(1, 6):
            if n_r[r] and n_r[r + 1]:
                d = (r + 1) * n_r[r + 1] / (r * n_r[r])
                self.discount[r] = min(1.0, max(0.01, d))
            else:
                self.discount[r] = 1.0
        self._alpha = {}

    def _d(self, count):
        """bigram 횟수 count의 할인 계수. 5회를 넘으면 할인하지 않으므로 1.0."""
        return self.discount[count] if count <= 5 else 1.0

    def p_unigram(self, word):
        """add-one 평활 unigram 확률 (c+1)/(T+V+1). 코퍼스에 없는 어절도 0이 되지 않는다."""
        return (self.unigrams.get(word, 0) + 1) / (self.total + self.v + 1)

    def _alpha_of(self, a):
        """문맥 a 뒤에서 본 적 없는 bigram에 나눠 줄 후퇴 가중치 α(a). 문맥마다 한 번만 계산해 저장해 둔다."""
        if a not in self._alpha:
            seen = self.following[a]
            ctx = self.context_total[a]
            numerator = 1.0 - sum(self._d(c) * c / ctx for _, c in seen)
            denominator = max(1.0 - sum(self.p_unigram(b) for b, _ in seen), TINY)
            self._alpha[a] = TINY if numerator <= 0 else numerator / denominator
        return self._alpha[a]

    def p(self, b, a):
        """p(b | a). 본 bigram은 할인한 비율, 못 본 bigram은 α(a)·P_uni(b), a 뒤에 어절이 온 적이 없으면 P_uni(b)."""
        count = self.bigrams.get((a, b), 0)
        if count > 0:
            return self._d(count) * count / self.context_total[a]
        if self.context_total.get(a, 0) > 0:
            return self._alpha_of(a) * self.p_unigram(b)
        return self.p_unigram(b)

    def log_prob(self, text):
        """(1/N) * sum(log p(w_k | w_{k-1})). 어절이 없는 문장이면 None."""
        tokens = tokenize(text)
        if not tokens:
            return None
        previous = START
        total = 0.0
        for word in tokens:
            total += math.log(self.p(word, previous))
            previous = word
        return total / len(tokens)

    def fm(self, candidate, reference):
        """길이로 정규화한 두 확률의 min/max. 로그로 계산한다."""
        lc, lr = self.log_prob(candidate), self.log_prob(reference)
        if lc is None or lr is None:
            return 0.0
        return math.exp(-abs(lc - lr))

    def oov_rates(self, candidate):
        """후보의 (미등록 어절 비율, 미등록 bigram 비율). 비어 있으면 (None, None)."""
        tokens = tokenize(candidate)
        if not tokens:
            return None, None
        words = sum(1 for w in tokens if w not in self.vocab)
        padded = [START] + tokens
        pairs = list(zip(padded, padded[1:]))
        unseen = sum(1 for pair in pairs if pair not in self.bigrams)
        return words / len(tokens), unseen / len(pairs)


class LSI:
    """코퍼스의 LSI 공간: tf-idf 어휘-문서 행렬, 절단 SVD, fold-in."""

    def __init__(self, sentences, dim=10):
        """코퍼스를 tf-idf 어휘×문서 행렬로 만들고, SVD로 앞 dim개 방향(projection)을 구한다."""
        docs = [tokenize(s) for s in sentences]
        docs = [d for d in docs if d]
        if not docs:
            raise ValueError("the corpus has no tokens")
        self.vocab = sorted({w for d in docs for w in d})
        self.index = {w: i for i, w in enumerate(self.vocab)}
        n = len(docs)
        df = Counter(w for d in docs for w in set(d))
        self.idf = np.array([math.log((1 + n) / (1 + df[w])) + 1 for w in self.vocab])
        matrix = np.zeros((len(self.vocab), n))
        for j, d in enumerate(docs):
            for w in d:
                matrix[self.index[w], j] += 1
        matrix *= self.idf[:, None]
        u, _, _ = np.linalg.svd(matrix, full_matrices=False)
        k = min(dim, n, len(self.vocab))
        self.projection = u[:, :k]

    def vector(self, text):
        """문장을 LSI 공간의 벡터로 바꾼다(어절 횟수 × idf에 projection을 곱함). 어휘 밖 어절은 무시한다."""
        counts = np.zeros(len(self.vocab))
        for w in tokenize(text):
            if w in self.index:
                counts[self.index[w]] += 1
        return (counts * self.idf) @ self.projection

    def am(self, candidate, reference):
        """AM. 두 문장 벡터의 코사인 유사도. 음수는 0으로 자르고, 벡터 크기가 1e-12 이하이면 0."""
        a, b = self.vector(candidate), self.vector(reference)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na <= TINY or nb <= TINY:
            return 0.0
        return max(0.0, float(a @ b) / (na * nb))


def ref_coverage(lm, references):
    """참조 문장의 어절(중복 포함) 중 코퍼스 어휘에 있는 비율."""
    tokens = [w for r in references for w in tokenize(r)]
    if not tokens:
        return None
    return sum(1 for w in tokens if w in lm.vocab) / len(tokens)
