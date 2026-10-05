"""역할 B 검증 사례와 언어모델·LSI 정의 (손으로 계산한 작은 사례)."""
import math
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

import numpy as np

from scorer import scoring_b
from scorer.dataio import InputError, read_outputs
from scorer.lm import LSI, TINY, BigramLM, ref_coverage, tokenize

from .helpers import dataset


@lru_cache(maxsize=1)
def corpus():
    return scoring_b.build_corpus(dataset())


@lru_cache(maxsize=1)
def scorer():
    return scoring_b.SentenceScorer(corpus())


class TinyLM(unittest.TestCase):
    """코퍼스: "a b"와 "a c". bigram (<s>,a)=2, (a,b)=1, (a,c)=1이고 n_1=2, n_2=1."""

    def setUp(self):
        self.lm = BigramLM(["a b", "a c"])

    def test_counts_and_discounts(self):
        self.assertEqual((self.lm.total, self.lm.v), (4, 3))
        self.assertEqual(self.lm.discount[1], 1.0)        # (1+1)*n_2 / (1*n_1) = 2*1/2
        self.assertEqual(self.lm.discount[2], 1.0)        # n_3 = 0

    def test_seen_bigrams(self):
        self.assertAlmostEqual(self.lm.p("a", "<s>"), 1.0)
        self.assertAlmostEqual(self.lm.p("b", "a"), 0.5)

    def test_unigram_is_add_one(self):
        self.assertAlmostEqual(self.lm.p_unigram("a"), 3 / 8)       # (2+1)/(4+3+1)
        self.assertAlmostEqual(self.lm.p_unigram("zzz"), 1 / 8)

    def test_backoff_with_exhausted_mass(self):
        # "a" 뒤의 확률 질량을 본 어절이 모두 쓰므로 분자가 0이고 alpha = 1e-12
        self.assertAlmostEqual(self.lm.p("a", "a") / (TINY * 3 / 8), 1.0)

    def test_context_never_followed_uses_unigram(self):
        self.assertAlmostEqual(self.lm.p("a", "b"), 3 / 8)          # "b"는 항상 문장의 마지막이다

    def test_sentence_probability_and_fm(self):
        self.assertAlmostEqual(self.lm.log_prob("a b"), math.log(0.5) / 2)
        self.assertAlmostEqual(self.lm.fm("a b", "a c"), 1.0)
        expected = math.exp(-abs(self.lm.log_prob("a a") - self.lm.log_prob("a b")))
        self.assertAlmostEqual(self.lm.fm("a a", "a b"), expected)
        self.assertLess(self.lm.fm("a a", "a b"), 1e-4)

    def test_oov_rates(self):
        self.assertEqual(self.lm.oov_rates("a z"), (0.5, 0.5))      # z는 미등록, (a,z)는 본 적 없음
        self.assertEqual(self.lm.oov_rates("a b"), (0.0, 0.0))
        self.assertEqual(self.lm.oov_rates("b a"), (0.0, 1.0))      # (<s>,b)와 (b,a)는 본 적 없음
        self.assertEqual(self.lm.oov_rates("   "), (None, None))

    def test_counts_above_five_are_not_discounted(self):
        lm = BigramLM(["a b"] * 7 + ["a c"] * 3 + ["a d", "a e"])
        self.assertAlmostEqual(lm.p("b", "a"), 7 / 12)               # r=7 > 5이므로 할인 없음, 문맥 합계 12

    def test_discount_above_one_is_clipped_to_one(self):
        lm = BigramLM(["a b", "a c", "a c", "d e", "d e", "d e"])
        # n_1=1, n_2=1 -> d_1 = 2*1/(1*1) = 2 -> 1.0
        self.assertEqual(lm.discount[1], 1.0)

    def test_discount_below_point_zero_one_is_clipped_up(self):
        sentences = ["w%d x%d" % (i, i) for i in range(300)] + ["p q", "p q"]
        lm = BigramLM(sentences)
        n1 = sum(1 for c in lm.bigrams.values() if c == 1)
        n2 = sum(1 for c in lm.bigrams.values() if c == 2)
        self.assertLess(2 * n2 / n1, 0.01)
        self.assertEqual(lm.discount[1], 0.01)

    def test_discount_between_the_bounds_and_leftover_mass(self):
        lm = BigramLM(["a b", "c d", "e f", "g h", "i j", "i j"])
        # (<s>,x) 4개와 (x,y) 4개는 한 번, (<s>,i)와 (i,j)는 두 번 나온다: n_1=8, n_2=2, d_1=0.5
        n1 = sum(1 for c in lm.bigrams.values() if c == 1)
        n2 = sum(1 for c in lm.bigrams.values() if c == 2)
        self.assertAlmostEqual(lm.discount[1], 2 * n2 / n1)
        # 문맥 "a": 뒤에 온 어절은 b 하나이고 c(a,b)=1
        d1 = lm.discount[1]
        p_b = lm.p_unigram("b")
        alpha = (1 - d1 * 1 / 1) / (1 - p_b)
        self.assertAlmostEqual(lm.p("zzz", "a"), alpha * lm.p_unigram("zzz"))
        self.assertAlmostEqual(lm.p("b", "a"), d1)

    def test_tokenize_is_plain_whitespace_split(self):
        self.assertEqual(tokenize("가  나\t다 라"), ["가", "나", "다", "라"])


class TinyLSI(unittest.TestCase):
    def test_identity_and_zero(self):
        lsi = LSI(["a b c", "a d", "e f g", "b e"], dim=3)
        self.assertAlmostEqual(lsi.am("a b c", "a b c"), 1.0)
        self.assertEqual(lsi.am("zzz", "a b c"), 0.0)             # 어휘 밖 어절뿐이므로 영벡터
        self.assertEqual(lsi.am("", "a b c"), 0.0)

    def test_negative_cosine_is_cut_to_zero(self):
        lsi = LSI(["a b", "a b", "c d", "c d", "a c"], dim=2)
        value = lsi.am("a b", "c d")
        self.assertGreaterEqual(value, 0.0)

    def test_dimension_is_limited_by_corpus(self):
        lsi = LSI(["a b", "a c"], dim=10)
        self.assertEqual(lsi.projection.shape[1], 2)


class RealCorpus(unittest.TestCase):
    def test_corpus_built_from_the_data(self):
        sentences = corpus()
        self.assertEqual(len(sentences), 32)
        self.assertEqual(len(set(sentences)), 32)
        self.assertFalse(set(sentences) & set(dataset().references()))
        self.assertTrue(all("\n" not in s for s in sentences))

    def test_same_corpus_from_the_other_file(self):
        data = dataset()
        from_a = []
        for case_id in data.ids:
            for m in data.user_json(data.row_a(case_id))["context"]:
                if m["role"] == "assistant" and m["content"] not in from_a:
                    from_a.append(m["content"])
        self.assertEqual(len(from_a), 38)
        self.assertEqual(set(from_a) - set(data.references()), set(corpus()))

    def test_corpus_file_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.txt"
            scoring_b.write_corpus(corpus(), path)
            self.assertEqual(scoring_b.read_corpus(path), corpus())

    def test_ref_coverage(self):
        self.assertAlmostEqual(ref_coverage(scorer().lm, dataset().references()), 0.4518, places=3)
        self.assertEqual(len(tokenize(" ".join(dataset().references()))), 1173)


class CasesB(unittest.TestCase):
    def setUp(self):
        self.data = dataset()
        self.refs = {i: self.data.reference(i) for i in self.data.ids}

    def test_1_same_as_reference(self):
        for case_id, ref in self.refs.items():
            s = scorer().score(ref, [ref])
            self.assertEqual(s["FM"], 1.0, case_id)
            self.assertAlmostEqual(s["AM"], 1.0, delta=1e-9, msg=case_id)

    def test_2_empty_string(self):
        s = scorer().score("", [self.refs[self.data.ids[0]]])
        self.assertEqual((s["FM"], s["AM"], s["oov_word"], s["oov_bigram"]), (0.0, 0.0, None, None))

    def test_3_no_line_in_output_file(self):
        rows = scoring_b.score_run(self.data, {}, scorer())
        self.assertTrue(all(r["missing"] and r["FM"] == 0 and r["AM"] == 0 for r in rows))

    def test_4_candidate_not_a_string(self):
        rows = scoring_b.score_run(self.data, {self.data.ids[0]: 123}, scorer())
        self.assertTrue(rows[0]["missing"])
        self.assertEqual((rows[0]["FM"], rows[0]["AM"]), (0.0, 0.0))
        self.assertIsNone(rows[0]["oov_word"])

    def test_5_last_word_removed(self):
        fm, am = [], []
        for case_id, ref in self.refs.items():
            s = scorer().score(" ".join(tokenize(ref)[:-1]), [ref])
            fm.append(s["FM"])
            am.append(s["AM"])
        self.assertTrue(all(x < 1 for x in fm))
        self.assertTrue(all(x <= 1 + 1e-9 for x in am))
        self.assertEqual(sum(1 for x in am if x < 1 - 1e-9), 69)
        self.assertEqual(sum(1 for x in am if abs(x - 1) < 1e-9), 31)
        self.assertEqual(sum(1 for x in am if x == 0), 3)

    def test_6_duplicate_id_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m2p_outputs.jsonl"
            path.write_text('{"id":"labeled100_001","candidate":"a"}\n{"id":"labeled100_001","candidate":"b"}\n',
                            encoding="utf-8")
            with self.assertRaises(InputError):
                read_outputs(path, "candidate", set(self.data.ids))

    def test_reference_as_candidate_oov_mean(self):
        oov = [scorer().score(r, [r])["oov_word"] for r in self.refs.values()]
        self.assertAlmostEqual(float(np.mean(oov)), 0.5456, places=3)

    def test_best_reference_is_used(self):
        refs = list(self.refs.values())[:2]
        s = scorer().score(refs[0], refs)
        self.assertEqual(s["FM"], 1.0)


if __name__ == "__main__":
    unittest.main()
