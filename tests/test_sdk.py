"""SDK(scorer.api)와 명령줄 방식이 같은 결과를 내는지, SDK가 파일을 만들지 않는지 확인한다."""
import json
import tempfile
import unittest
from pathlib import Path

import scorer
from scorer import Evaluator, RunResult, evaluate, pipeline, scoring_b
from scorer.dataio import InputError, read_outputs
from scorer.store import DiskStore, MemoryStore

from .helpers import ROOT, config, dataset, fingerprint, project_state
from .mockdata import mock_a, mock_b, write_mock_run
from .test_pipeline import SETTINGS, Workspace


def outputs_of(data, quality, seed):
    """mockdata가 만드는 출력을 SDK에 넘길 dict 두 개로 바꾼다. (같은 값이 파일로도 쓰인다.)"""
    a = {row["id"]: row["raw_output"] for row in mock_a(data, quality, seed)}
    b = {row["id"]: row["candidate"] for row in mock_b(data, quality, seed + 1)}
    return a, b


class Workbench(Workspace):
    """임시 폴더에 코퍼스 파일을 미리 만들어 두는 작업 공간. 두 방식이 같은 코퍼스를 쓰게 한다."""

    def __init__(self, tmp, **kwargs):
        super().__init__(tmp, **kwargs)
        scoring_b.write_corpus(scoring_b.build_corpus(self.data), self.config.corpus_path)
        self.evaluator = Evaluator(self.config)


class SameAsFiles(unittest.TestCase):
    """같은 출력을 파일 방식과 SDK 방식으로 채점하면 결과가 정확히 같다."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ws = Workbench(cls.tmp.name)
        cls.plan = [("base", 0.5, 1)] + [("ft_seed%d" % s, 0.93, 30 + i) for i, s in enumerate((42, 52, 62, 72, 82))]
        cls.sdk = {}
        for run_id, quality, seed in cls.plan:
            a, b = outputs_of(cls.ws.data, quality, seed)
            cls.sdk[run_id] = cls.ws.evaluator.score(run_id, a, b, model_id="mock", decoding=SETTINGS)
            write_mock_run(cls.ws.config.runs_dir, run_id, cls.ws.data, quality, seed)
            pipeline.score_run_folder(cls.ws.config, cls.ws.data, run_id, model_id="mock", decoding=SETTINGS)
        cls.disk = DiskStore(cls.ws.config.results_dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_result_file_is_identical(self):
        names = ["m1v_scores.jsonl", "m1v_summary.json", "m2p_scores.jsonl", "m2p_summary.json", "run_meta.json"]
        for run_id, _, _ in self.plan:
            self.assertEqual(sorted(self.sdk[run_id].artifacts), sorted(names))
            for name in names:
                on_disk = (self.disk.read_rows(run_id, name) if name.endswith(".jsonl")
                           else self.disk.read_json(run_id, name))
                self.assertEqual(self.sdk[run_id].artifacts[name], on_disk, (run_id, name))

    def test_comparison_is_identical(self):
        results = [self.sdk[run_id] for run_id, _, _ in self.plan]
        self.assertEqual(self.ws.evaluator.compare(results), pipeline.compare_runs(self.ws.config))
        self.assertEqual(self.ws.evaluator.compare(), pipeline.compare_runs(self.ws.config))   # 채점한 모든 실행

    def test_report_text_is_identical(self):
        from scorer import report
        results = [self.sdk[run_id] for run_id, _, _ in self.plan]
        in_memory = self.ws.evaluator.report(results)
        on_disk = report.build_report(self.ws.config, pipeline.compare_runs(self.ws.config), self.ws.data)
        self.assertEqual(in_memory, on_disk)
        self.assertTrue(in_memory.startswith("| 축 | 지표 |"))

    def test_score_folder_matches_too(self):
        result = self.ws.evaluator.score_folder("ft_seed42", model_id="mock", decoding=SETTINGS)
        self.assertEqual(result.artifacts, self.sdk["ft_seed42"].artifacts)

    def test_result_accessors(self):
        result = self.sdk["base"]
        self.assertIsInstance(result, RunResult)
        self.assertEqual(result.meta["model_id"], "mock")
        self.assertEqual(result.meta["decoding"], SETTINGS)
        self.assertEqual(len(result.a_scores), 100)
        self.assertEqual(len(result.b_scores), 100)
        self.assertEqual(result.a_summary["n_cases"], 100)
        self.assertIn("fm_mean", result.b_summary)


class NoSideEffects(unittest.TestCase):
    def test_sdk_writes_nothing_and_does_not_touch_the_data(self):
        before = fingerprint(config().data_dir)
        project_before = project_state()
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)          # 이 작업 공간에는 코퍼스 파일도 없다
            ev = Evaluator(ws.config)
            a, b = outputs_of(ws.data, 0.9, 1)
            result = ev.score("base", a, b)
            ev.score("ft_seed42", a, b)
            ev.compare()
            ev.report()
            self.assertFalse(ws.config.results_dir.exists())      # results/도 만들지 않는다
            self.assertFalse(ws.config.corpus_path.exists())      # 코퍼스 파일도 만들지 않는다
            self.assertFalse(ws.config.runs_dir.exists())
            self.assertTrue(any("built in memory" in w for w in result.warnings))
        self.assertEqual(fingerprint(config().data_dir), before)
        self.assertEqual(project_state(), project_before)     # 프로젝트의 runs/, results/도 그대로

    def test_save_writes_exactly_one_runs_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workbench(tmp)
            a, b = outputs_of(ws.data, 0.9, 1)
            result = ws.evaluator.score("base", a, b)
            names = ws.evaluator.save(result)
            written = sorted(p.name for p in (ws.config.results_dir / "base").iterdir())
        self.assertEqual(sorted(names), written)
        self.assertEqual(written, ["m1v_scores.jsonl", "m1v_summary.json", "m2p_scores.jsonl",
                                   "m2p_summary.json", "run_meta.json"])


class InputHandling(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Workbench(self.tmp.name)
        self.ev = self.ws.evaluator
        self.ids = self.ws.data.ids

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_role_only(self):
        a, b = outputs_of(self.ws.data, 1.0, 0)
        only_a = self.ev.score("base", outputs_a=a)
        only_b = self.ev.score("ft_seed42", outputs_b=b)
        self.assertEqual((only_a.meta["hash_scope"], only_a.b_scores), ("A", None))
        self.assertEqual((only_b.meta["hash_scope"], only_b.a_scores), ("B", None))

    def test_nothing_to_score_is_an_error(self):
        with self.assertRaises(InputError):
            self.ev.score("base")

    def test_missing_unknown_and_non_string_outputs(self):
        a, b = outputs_of(self.ws.data, 1.0, 0)
        a = {k: v for k, v in list(a.items())[:90]}
        a["not-a-case"] = "x"
        a[5] = "x"
        result = self.ev.score("base", outputs_a=a, outputs_b={self.ids[0]: 123})
        self.assertEqual(result.a_summary["diagnostics"]["missing_outputs"], 10)
        self.assertEqual(result.b_summary["missing_outputs"], 100)
        self.assertEqual(sum("was ignored" in w for w in result.warnings), 2)
        self.assertTrue(any("10 of 100" in w for w in result.warnings))

    def test_wrong_types(self):
        for bad in ([("id", "x")], "text", 5):
            with self.assertRaises(InputError):
                self.ev.score("base", outputs_a=bad)
        a, _ = outputs_of(self.ws.data, 1.0, 0)
        with self.assertRaises(InputError):
            self.ev.score("base", outputs_a=a, decoding=["json_mode"])

    def test_decoding_is_recorded_as_given_or_warned(self):
        a, _ = outputs_of(self.ws.data, 1.0, 0)
        given = self.ev.score("base", outputs_a=a, decoding=SETTINGS)
        absent = self.ev.score("ft_seed42", outputs_a=a)
        self.assertEqual(given.meta["decoding"], SETTINGS)
        self.assertIsNone(absent.meta["decoding"])
        self.assertTrue(any("decoding settings were not provided" in w for w in absent.warnings))
        decoding = self.ev.compare([given, absent])["decoding"]
        self.assertIsNone(decoding["consistent"])
        self.assertEqual(decoding["missing"], ["ft_seed42"])

    def test_different_settings_between_runs_are_flagged(self):
        a, _ = outputs_of(self.ws.data, 1.0, 0)
        base = self.ev.score("base", outputs_a=a, decoding=SETTINGS)
        ft = self.ev.score("ft_seed42", outputs_a=a, decoding=dict(SETTINGS, json_mode_m1v=True))
        decoding = self.ev.compare([base, ft])["decoding"]
        self.assertFalse(decoding["consistent"])
        self.assertEqual([d["key"] for d in decoding["differences"]], ["json_mode_m1v"])

    def test_run_ids_must_follow_the_convention_when_comparing(self):
        a, _ = outputs_of(self.ws.data, 1.0, 0)
        odd = self.ev.score("my-run", outputs_a=a)
        with self.assertRaises(InputError):
            self.ev.compare([odd])
        # 이름이 달라도 채점 자체는 된다
        self.assertEqual(odd.a_summary["value_accuracy"], 1.0)

    def test_a_rescored_run_replaces_the_earlier_one(self):
        good, _ = outputs_of(self.ws.data, 1.0, 0)
        bad, _ = outputs_of(self.ws.data, 0.0, 0)
        self.ev.score("base", outputs_a=good)
        self.ev.score("base", outputs_a=bad)
        self.assertLess(self.ev.compare()["a"]["base"]["value_accuracy"], 0.5)

    def test_sentence_scorer_is_built_once(self):
        _, b = outputs_of(self.ws.data, 1.0, 0)
        self.ev.score("base", outputs_b=b)
        first = self.ev._sentence
        self.ev.score("ft_seed42", outputs_b=b)
        self.assertIs(self.ev._sentence, first)

    def test_only_one_side_cannot_be_compared(self):
        a, _ = outputs_of(self.ws.data, 1.0, 0)
        base = self.ev.score("base", outputs_a=a)
        comparison = self.ev.compare([base])
        self.assertEqual(comparison["a"]["mode"], "base_only")
        self.assertEqual(comparison["a"]["overall"], "미판정")


class Packaging(unittest.TestCase):
    def test_top_level_names(self):
        self.assertIs(scorer.Evaluator, Evaluator)
        self.assertIs(scorer.evaluate, evaluate)
        self.assertIs(scorer.RunResult, RunResult)

    def test_evaluate_shortcut(self):
        ws_config = ROOT / "config.json"
        project_before = project_state()
        a = {row["id"]: row["raw_output"] for row in mock_a(dataset(), 1.0, 0)}
        result = evaluate(a, config=ws_config, run_id="base")
        self.assertEqual(result.a_summary["value_accuracy"], 1.0)
        self.assertEqual(project_state(), project_before)      # 진짜 설정으로 불러도 아무것도 쓰지 않는다

    def test_data_check_failure_stops_the_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
            broken = Path(tmp) / "data"
            broken.mkdir()
            for name in ("extraction.jsonl", "response.jsonl"):
                lines = (config().data_dir / name).read_text(encoding="utf-8").splitlines()
                (broken / name).write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")   # 마지막 사례를 뺀다
            # 두 파일에서 같은 사례를 뺐으므로 길이만 99건이 되고 데이터 검사는 통과한다
            raw["paths"] = {"data_dir": str(broken), "runs_dir": "runs", "results_dir": "results"}
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            ev = Evaluator(path)
            self.assertEqual(len(ev.data.ids), 99)
            # 한쪽 파일만 줄이면 id 집합이 달라져 검사에 걸린다
            lines = (broken / "response.jsonl").read_text(encoding="utf-8").splitlines()
            (broken / "response.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaises(InputError):
                Evaluator(path)
            self.assertEqual(len(Evaluator(path, check=False).data.ids), 99)


if __name__ == "__main__":
    unittest.main()
