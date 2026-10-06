"""neural 문장 점수 백엔드 검증. 실제 모델은 내려받지 않고, 가짜 모델(토큰 확률과 벡터를 직접 정하는 객체)로 계산식과
파이프라인 연결을 확인한다. torch와 transformers가 있으면 작은 무작위 모델을 임시 폴더에 만들어 실제 불러오기 경로도
확인한다(모델 내려받기 없음).
"""
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scorer import Evaluator, neural, pipeline
from scorer.dataio import InputError
from scorer.neural import NeuralEmbedding, NeuralLM, NeuralSentenceScorer

from .helpers import ROOT, config
from .test_pipeline import Workspace

SPEC = {"type": "neural", "device": "cpu",
        "lm": {"model": "fake-lm", "revision": "r1", "dtype": "float32"},
        "embedder": {"model": "fake-emb", "revision": "r2", "pooling": "cls"}}


class FakeLM:
    """어절마다 -(1 + 글자 수 % 5)/2의 로그확률을 주는 가짜 언어모델. 같은 입력은 항상 같은 값이다."""

    def token_logprobs(self, text):
        return [-(1 + len(tok) % 5) / 2 for tok in text.split()]


class FakeEmbedder:
    """글자를 8칸에 세어 담는 가짜 임베딩. 같은 문장은 같은 벡터이다."""

    def encode(self, text):
        vector = np.zeros(8)
        for ch in text:
            vector[ord(ch) % 8] += 1
        return vector


def fake_scorer(spec=SPEC):
    return NeuralSentenceScorer(NeuralLM(FakeLM()), NeuralEmbedding(FakeEmbedder()), neural.parse_spec(spec))


def patched_build():
    """neural.build_scorer를 가짜 점수기를 만드는 함수로 바꾼다(모델을 내려받지 않는다)."""
    return mock.patch.object(neural, "build_scorer", side_effect=lambda spec: fake_scorer(spec))


class Formulas(unittest.TestCase):
    def test_fm_is_exp_of_minus_the_difference_of_mean_logprobs(self):
        class Fixed:
            table = {"a": [-1.0, -3.0], "b": [-1.0], "c": []}

            def token_logprobs(self, text):
                return self.table[text]

        lm = NeuralLM(Fixed())
        self.assertAlmostEqual(lm.log_prob("a"), -2.0)
        self.assertAlmostEqual(lm.fm("a", "b"), math.exp(-1.0))        # |-2 - (-1)| = 1
        self.assertAlmostEqual(lm.fm("a", "a"), 1.0)
        self.assertIsNone(lm.log_prob("c"))
        self.assertEqual(lm.fm("c", "a"), 0.0)                          # 토큰이 없으면 0
        self.assertEqual(lm.fm("a", "c"), 0.0)

    def test_lm_asks_the_model_once_per_sentence(self):
        class Counting:
            calls = 0

            def token_logprobs(self, text):
                Counting.calls += 1
                return [-1.0]

        lm = NeuralLM(Counting())
        lm.fm("x", "y")
        lm.fm("x", "y")
        lm.fm("x", "x")
        self.assertEqual(Counting.calls, 2)

    def test_am_is_cosine_clipped_at_zero(self):
        vectors = {"same": [1.0, 0.0], "same2": [2.0, 0.0], "ortho": [0.0, 1.0], "opposite": [-1.0, 0.0],
                   "zero": [0.0, 0.0], "diag": [1.0, 1.0]}

        class Table:
            def encode(self, text):
                return vectors[text]

        emb = NeuralEmbedding(Table())
        self.assertAlmostEqual(emb.am("same", "same2"), 1.0)
        self.assertAlmostEqual(emb.am("same", "ortho"), 0.0)
        self.assertEqual(emb.am("same", "opposite"), 0.0)              # 음수는 0
        self.assertEqual(emb.am("zero", "same"), 0.0)                  # 영벡터는 0
        self.assertAlmostEqual(emb.am("same", "diag"), 1 / math.sqrt(2))

    def test_scorer_returns_the_same_keys_as_the_ngram_scorer(self):
        scorer = fake_scorer()
        score = scorer.score("오늘 기분이 어떠세요", ["오늘 기분이 어떠세요"])
        self.assertEqual(list(score), ["FM", "AM", "oov_word", "oov_bigram"])
        self.assertAlmostEqual(score["FM"], 1.0)
        self.assertAlmostEqual(score["AM"], 1.0)
        self.assertEqual((score["oov_word"], score["oov_bigram"]), (None, None))
        for bad in (None, 5, "", "   "):
            self.assertEqual(scorer.score(bad, ["x"]), {"FM": 0.0, "AM": 0.0, "oov_word": None, "oov_bigram": None})

    def test_best_of_several_references_is_used(self):
        scorer = fake_scorer()
        both = scorer.score("가나 다라마", ["가나 다라마", "전혀 다른 문장입니다 정말로"])
        self.assertAlmostEqual(both["FM"], 1.0)


class Spec(unittest.TestCase):
    def test_parse_fills_defaults(self):
        spec = neural.parse_spec({"type": "neural", "lm": {"model": "m"}, "embedder": {"model": "e"}})
        self.assertEqual(spec["device"], "auto")
        self.assertEqual(spec["lm"], {"model": "m", "revision": None, "dtype": "auto"})
        self.assertEqual(spec["embedder"]["pooling"], "cls")
        self.assertEqual(spec["embedder"]["max_length"], 512)

    def test_bad_specs_are_input_errors(self):
        for bad in (None, {"type": "neural"}, {"lm": {"model": "m"}},
                    {"lm": {"model": "m"}, "embedder": {}},
                    {"lm": {"model": ""}, "embedder": {"model": "e"}},
                    {"lm": {"model": "m"}, "embedder": {"model": "e", "pooling": "max"}}):
            with self.assertRaises(InputError, msg=bad):
                neural.parse_spec(bad)

    def test_missing_torch_is_an_input_error(self):
        with mock.patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(InputError) as ctx:
                neural.build_scorer(SPEC)
        self.assertIn("torch", str(ctx.exception))


class DtypeChoice(unittest.TestCase):
    """GPU가 없는 환경에서도 가짜 torch로 정밀도 선택 규칙을 확인한다."""

    @staticmethod
    def fake_torch(capability_major):
        return types.SimpleNamespace(
            float32="float32", float16="float16", bfloat16="bfloat16",
            cuda=types.SimpleNamespace(get_device_capability=lambda device: (capability_major, 0)))

    def test_auto_on_cpu_is_float32(self):
        self.assertEqual(neural._dtype(self.fake_torch(8), "auto", "cpu"), "float32")

    def test_auto_on_an_old_gpu_is_float16_not_bfloat16(self):
        # Colab 무료의 T4는 연산 능력 7.5라서 bfloat16을 기본으로 지원하지 않는다
        self.assertEqual(neural._dtype(self.fake_torch(7), "auto", "cuda"), "float16")

    def test_auto_on_a_new_gpu_is_bfloat16(self):
        for device in ("cuda", "cuda:0"):
            self.assertEqual(neural._dtype(self.fake_torch(8), "auto", device), "bfloat16")
            self.assertEqual(neural._dtype(self.fake_torch(9), "auto", device), "bfloat16")

    def test_the_gpu_is_asked_about_the_device_that_was_chosen(self):
        asked = []
        torch = self.fake_torch(8)
        torch.cuda.get_device_capability = lambda device: asked.append(device) or (8, 0)
        neural._dtype(torch, "auto", "cuda:1")
        self.assertEqual(asked, ["cuda:1"])


class DtypeKeyword(unittest.TestCase):
    """from_pretrained에 dtype을 넘기는 인자 이름은 transformers 4.56을 경계로 바뀐다."""

    def kwargs(self, version):
        return neural._dtype_kwargs(types.SimpleNamespace(__version__=version), "X")

    def test_new_versions_use_dtype(self):
        for version in ("4.56.0", "4.56.2", "4.57.1", "5.0.0", "5.18.0", "4.56.0.dev0", "5.0.0rc1"):
            self.assertEqual(self.kwargs(version), {"dtype": "X"}, version)

    def test_old_versions_use_torch_dtype(self):
        for version in ("4.55.4", "4.46.0", "4.9.2", "3.0.0"):
            self.assertEqual(self.kwargs(version), {"torch_dtype": "X"}, version)

    def test_an_unreadable_version_uses_the_new_name(self):
        for version in ("", "unknown", "5"):
            self.assertEqual(self.kwargs(version), {"dtype": "X"}, repr(version))
        self.assertEqual(neural._dtype_kwargs(types.SimpleNamespace(), "X"), {"dtype": "X"})


class ShippedNeuralConfig(unittest.TestCase):
    """저장소에 들어 있는 config.neural.json이 재현 가능하고 목표 환경에 맞는 상태인지 확인한다."""

    @classmethod
    def setUpClass(cls):
        raw = json.loads((ROOT / "config.neural.json").read_text(encoding="utf-8"))
        cls.spec = neural.parse_spec(raw["sentence_scorer"])

    def test_models_are_pinned_to_a_revision(self):
        # revision이 비어 있으면 같은 모델 이름이어도 나중에 받은 것이 달라져 점수를 다시 만들 수 없다
        for part in ("lm", "embedder"):
            revision = self.spec[part]["revision"]
            self.assertIsInstance(revision, str, part)
            self.assertRegex(revision, r"^[0-9a-f]{40}$", part)

    def test_precision_is_explicit_and_safe_on_every_gpu(self):
        # auto에 맡기면 GPU 종류에 따라 정밀도가 달라지고, float16은 일부 모델에서 수치가 불안정할 수 있다
        for part in ("lm", "embedder"):
            self.assertEqual(self.spec[part]["dtype"], "float32", part)

    def test_it_does_not_ask_for_a_seven_billion_parameter_model(self):
        # 8~16GB 안에서, Colab 무료의 T4에서도 돌아가는 크기여야 한다
        self.assertNotRegex(self.spec["lm"]["model"].lower(), r"(7b|8b|9b|13b|14b|32b|70b)")

    def test_it_only_differs_from_the_main_config_in_the_scorer_and_the_results_folder(self):
        main = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        neural_config = json.loads((ROOT / "config.neural.json").read_text(encoding="utf-8"))
        for key in ("seeds", "ci", "thresholds", "rule_name"):
            self.assertEqual(main.get(key), neural_config.get(key), key)
        self.assertNotEqual(main["paths"]["results_dir"], neural_config["paths"]["results_dir"])


class InPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ws = Workspace(cls.tmp.name)
        cls.ws.config.sentence_scorer = SPEC
        with patched_build():
            cls.info = cls.ws.run("base", 0.5, seed=1, roles=("B",))
            cls.ws.run("ft_seed42", 0.95, seed=2, roles=("B",))
            cls.comparison, cls.text = cls.ws.report_text()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_summary_records_the_models_and_has_no_corpus_fields(self):
        b = pipeline.read_json(self.ws.config.results_dir / "base" / "m2p_summary.json")
        self.assertEqual(b["scorer"]["type"], "neural")
        self.assertEqual(b["scorer"]["lm"]["model"], "fake-lm")
        self.assertEqual(b["scorer"]["embedder"]["model"], "fake-emb")
        self.assertIsNone(b["corpus_sentences"])
        self.assertFalse(b["corpus_below_minimum"])
        self.assertIsNone(b["ref_coverage"])
        self.assertIsNone(b["oov_word_mean"])
        self.assertIsNotNone(b["fm_mean"])
        self.assertIsNotNone(b["am_mean"])

    def test_no_corpus_file_is_written_and_no_corpus_warning(self):
        self.assertFalse(self.ws.config.corpus_path.exists())
        self.assertFalse(any("corpus" in w for w in self.info["warnings"]))

    def test_run_meta_records_the_scorer(self):
        meta = pipeline.read_json(self.ws.config.results_dir / "base" / "run_meta.json")
        self.assertEqual(meta["sentence_scorer"], neural.parse_spec(SPEC))

    def test_score_rows_keep_the_same_columns(self):
        from scorer.store import DiskStore
        rows = DiskStore(self.ws.config.results_dir).read_rows("base", "m2p_scores.jsonl")
        self.assertEqual(list(rows[0]), ["id", "missing", "FM", "AM", "oov_word", "oov_bigram", "completion_claim"])

    def test_same_scorer_in_both_runs_is_consistent(self):
        self.assertTrue(self.comparison["sentence_scorer"]["consistent"])
        self.assertNotIn("문장 점수 방식이 다릅니다", self.text)

    def test_report_names_the_models_instead_of_the_corpus(self):
        self.assertIn("문장 점수 방식: neural", self.text)
        self.assertIn("fake-lm", self.text)
        self.assertIn("fake-emb", self.text)
        self.assertNotIn("코퍼스 규모 미달", self.text)
        self.assertNotIn("ref_coverage", self.text)

    def test_mixing_ngram_and_neural_runs_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            ws.run("base", 0.5, seed=1, roles=("B",))                  # 기본 ngram
            ws.config.sentence_scorer = SPEC
            with patched_build():
                ws.run("ft_seed42", 0.9, seed=2, roles=("B",))
            comparison, text = ws.report_text()
        self.assertFalse(comparison["sentence_scorer"]["consistent"])
        self.assertIn("실행마다 문장 점수 방식이 다릅니다", text)

    def test_unknown_type_is_an_input_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            ws.config.sentence_scorer = {"type": "gpt"}
            with self.assertRaises(InputError):
                ws.run("base", 0.5, seed=1, roles=("B",))

    def test_role_a_only_runs_do_not_load_any_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            ws.config.sentence_scorer = SPEC
            with mock.patch.object(neural, "build_scorer", side_effect=AssertionError("loaded")):
                ws.run("base", 0.5, seed=1, roles=("A",))
            meta = pipeline.read_json(ws.config.results_dir / "base" / "run_meta.json")
        self.assertIsNone(meta["sentence_scorer"])


class InSdk(unittest.TestCase):
    def test_evaluator_uses_the_neural_scorer_without_writing_files(self):
        from .mockdata import mock_b
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            ws.config.sentence_scorer = SPEC
            ev = Evaluator(ws.config)
            outputs_b = {row["id"]: row["candidate"] for row in mock_b(ws.data, 0.7, 3)}
            with patched_build() as build:
                first = ev.score("base", None, outputs_b)
                second = ev.score("ft_seed42", None, outputs_b)
            self.assertEqual(build.call_count, 1)                       # 모델은 한 번만 불러온다
            self.assertEqual(first.b_summary["scorer"]["type"], "neural")
            self.assertEqual(first.b_scores, second.b_scores)
            self.assertFalse(ws.config.corpus_path.exists())
            self.assertFalse(ws.config.results_dir.exists())


try:
    import tokenizers
    import torch
    import transformers
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch, transformers, tokenizers가 없으면 건너뜀")
class RealTorchPath(unittest.TestCase):
    """작은 무작위 모델을 임시 폴더에 만들어 TorchCausalLM과 TorchEmbedder를 실제로 불러온다(내려받기 없음).

    점수 값 자체가 아니라 불러오기, 토큰 수, 확률의 범위, 같은 입력의 같은 출력, 벡터 크기를 확인한다.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        words = "오늘 기분이 어떠세요 좋습니다 네 아니요 물 한 잔 마셨어요".split()
        vocab = {"[UNK]": 0, "[BOS]": 1, "[PAD]": 2}
        for w in words:
            vocab[w] = len(vocab)
        tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", bos_token="[BOS]",
                                                    pad_token="[PAD]")
        cls.lm_dir = Path(cls.tmp.name) / "lm"
        cls.emb_dir = Path(cls.tmp.name) / "emb"
        torch.manual_seed(0)
        lm_config = transformers.GPT2Config(vocab_size=len(vocab), n_embd=16, n_layer=1, n_head=2, n_positions=32)
        transformers.GPT2LMHeadModel(lm_config).save_pretrained(cls.lm_dir)
        fast.save_pretrained(cls.lm_dir)
        emb_config = transformers.BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                                             num_attention_heads=2, intermediate_size=32)
        transformers.BertModel(emb_config).save_pretrained(cls.emb_dir)
        fast.save_pretrained(cls.emb_dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def spec(self, pooling):
        return neural.parse_spec({"device": "cpu", "lm": {"model": str(self.lm_dir), "dtype": "float32"},
                                  "embedder": {"model": str(self.emb_dir), "dtype": "float32", "pooling": pooling}})

    def test_token_logprobs(self):
        lm = neural.TorchCausalLM(self.spec("cls")["lm"], "cpu")
        logprobs = lm.token_logprobs("오늘 기분이 어떠세요")
        self.assertEqual(len(logprobs), 3)
        self.assertTrue(all(lp <= 0.0 for lp in logprobs))
        self.assertEqual(logprobs, lm.token_logprobs("오늘 기분이 어떠세요"))
        self.assertEqual(lm.token_logprobs(""), [])

    def test_embedding_is_a_unit_vector_for_both_poolings(self):
        for pooling in ("cls", "mean"):
            emb = neural.TorchEmbedder(self.spec(pooling)["embedder"], "cpu")
            vector = emb.encode("오늘 기분이 어떠세요")
            self.assertEqual(vector.shape, (16,))
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)
            self.assertTrue(np.allclose(vector, emb.encode("오늘 기분이 어떠세요")))

    def test_build_scorer_end_to_end(self):
        scorer = neural.build_scorer(self.spec("cls"))
        same = scorer.score("오늘 기분이 어떠세요", ["오늘 기분이 어떠세요"])
        self.assertAlmostEqual(same["FM"], 1.0)
        self.assertAlmostEqual(same["AM"], 1.0, places=5)
        other = scorer.score("물 한 잔 마셨어요", ["오늘 기분이 어떠세요"])
        self.assertTrue(0.0 <= other["FM"] <= 1.0 and 0.0 <= other["AM"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
