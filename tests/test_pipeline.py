"""임시 폴더에서 처음부터 끝까지: 가짜 출력 -> 채점 -> 비교 -> 리포트.

실제 데이터셋은 읽기만 하고(파일이 그대로인지 검사한다), 프로젝트의 runs/와 results/ 폴더에는
아무것도 쓰지 않는다.
"""
import contextlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scorer import __main__ as cli
from scorer import pipeline, report
from scorer.dataio import InputError

from .helpers import ROOT, config, dataset, fingerprint, project_state
from .mockdata import write_mock_run


class Workspace:
    """자체 설정과 runs/, results/를 가진 임시 프로젝트 루트."""

    def __init__(self, tmp, ci_b=200, seeds=(42, 52, 62, 72, 82)):
        self.root = Path(tmp)
        raw = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        raw["paths"] = {"data_dir": str(config().data_dir), "runs_dir": "runs", "results_dir": "results"}
        raw["ci"]["B"] = ci_b
        raw["seeds"] = list(seeds)
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(raw), encoding="utf-8")
        self.config = pipeline.Config(self.config_path)
        self.data = dataset()

    def run(self, run_id, quality, seed, roles=("A", "B"), decoding=None):
        write_mock_run(self.config.runs_dir, run_id, self.data, quality, seed, roles)
        return pipeline.score_run_folder(self.config, self.data, run_id, model_id="mock", decoding=decoding)

    def report_text(self):
        comparison = pipeline.compare_runs(self.config)
        pipeline.write_json(self.config.results_dir / "comparison.json", comparison)
        report.write_report(self.config, comparison, self.data)
        return comparison, (self.config.results_dir / "report.md").read_text(encoding="utf-8")


class FullRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.before = fingerprint(config().data_dir)
        cls.project_before = project_state()
        cls.ws = Workspace(cls.tmp.name)
        cls.ws.run("base", 0.5, seed=1)
        for i, seed in enumerate((42, 52, 62, 72, 82)):
            cls.ws.run("ft_seed%d" % seed, 1.0 if i == 0 else 0.95, seed=10 + i)
        cls.comparison, cls.text = cls.ws.report_text()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_dataset_is_untouched(self):
        self.assertEqual(fingerprint(config().data_dir), self.before)

    def test_project_runs_and_results_are_unchanged(self):
        """테스트가 프로젝트의 runs/, results/에 쓰지 않았다(전후 비교. 실제 파일이 있어도 틀리지 않는다)."""
        self.assertEqual(project_state(), self.project_before)

    def test_inputs_and_generated_files_are_in_separate_folders(self):
        runs = {p.name for p in self.ws.config.runs_dir.rglob("*") if p.is_file()}
        self.assertEqual(runs, {"m1v_outputs.jsonl", "m2p_outputs.jsonl"})
        results = {p.name for p in self.ws.config.results_dir.rglob("*") if p.is_file()}
        self.assertEqual(results, {"m1v_scores.jsonl", "m2p_scores.jsonl", "m1v_summary.json", "m2p_summary.json",
                                   "run_meta.json", "corpus.txt", "comparison.json", "report.md"})

    def test_score_files_have_the_documented_keys(self):
        rows = pipeline._read_rows(self.ws.config.results_dir / "base" / "m1v_scores.jsonl")
        self.assertEqual(len(rows), 100)
        self.assertEqual(list(rows[0]), ["id", "missing", "parse_ok", "schema_ok", "values_ok", "correct",
                                         "core_correct", "evidence_ok", "gold_oos", "pred_oos", "category"])
        rows = pipeline._read_rows(self.ws.config.results_dir / "base" / "m2p_scores.jsonl")
        self.assertEqual(list(rows[0]), ["id", "missing", "FM", "AM", "oov_word", "oov_bigram", "completion_claim"])

    def test_perfect_run_scores_perfectly(self):
        summary = pipeline.read_json(self.ws.config.results_dir / "ft_seed42" / "m1v_summary.json")
        self.assertEqual(summary["value_accuracy"], 1.0)
        self.assertEqual(summary["f1_oos"], 1.0)
        self.assertEqual(summary["schema_accuracy"], 1.0)
        self.assertEqual(summary["diagnostics"], {"parse_failures": 0, "schema_failures": 0, "values_violations": 0,
                                                  "value_cases_judged_out_of_scope": 0, "missing_outputs": 0})
        b = pipeline.read_json(self.ws.config.results_dir / "ft_seed42" / "m2p_summary.json")
        self.assertEqual(b["fm_mean"], 1.0)
        self.assertAlmostEqual(b["am_mean"], 1.0, delta=1e-9)
        self.assertTrue(b["corpus_below_minimum"])
        self.assertEqual(b["corpus_sentences"], 32)

    def test_damaged_run_scores_lower(self):
        s = pipeline.read_json(self.ws.config.results_dir / "base" / "m1v_summary.json")
        self.assertLess(s["value_accuracy"], 0.8)
        self.assertGreater(s["diagnostics"]["parse_failures"] + s["diagnostics"]["schema_failures"]
                           + s["diagnostics"]["values_violations"] + s["diagnostics"]["value_cases_judged_out_of_scope"], 0)

    def test_new_auxiliary_metrics_reach_summaries_and_report(self):
        """근거 유효율과 계획 위반 표현이 점수 파일, 요약, 리포트에 모두 나온다."""
        results = self.ws.config.results_dir
        perfect_a = pipeline.read_json(results / "ft_seed42" / "m1v_summary.json")["evidence"]
        self.assertEqual((perfect_a["output_valid_rate"], perfect_a["item_valid_rate"]), (1.0, 1.0))
        self.assertEqual(pipeline.read_json(results / "ft_seed42" / "m2p_summary.json")["completion_claims"], 0)
        damaged_a = pipeline.read_json(results / "base" / "m1v_summary.json")["evidence"]
        self.assertLess(damaged_a["output_valid_rate"], 1.0)
        self.assertLess(damaged_a["item_valid_rate"], 1.0)
        self.assertGreater(pipeline.read_json(results / "base" / "m2p_summary.json")["completion_claims"], 0)
        rows = pipeline._read_rows(results / "base" / "m1v_scores.jsonl")
        self.assertTrue(any(r["evidence_ok"] is False for r in rows))
        self.assertTrue(any(r["evidence_ok"] is None for r in rows))         # 읽을 수 없는 출력은 null
        self.assertTrue(any(r["completion_claim"] for r in pipeline._read_rows(results / "base" / "m2p_scores.jsonl")))
        text = self.text
        self.assertIn("### 근거 유효율과 계획 위반 표현 (실행별, 판정에 쓰지 않음)", text)
        self.assertIn("- `base`: A 근거 유효 출력", text)
        self.assertIn("B 계획 위반 표현", text)

    def test_rule_name_line_is_left_out_when_the_config_has_none(self):
        """규칙 이름은 설정 값이라서, 설정에 없으면 그 줄이 리포트에 나오지 않는다."""
        self.ws.config.rule_name = None
        try:
            text = report.build_report(self.ws.config, self.comparison, self.ws.data)
        finally:
            self.ws.config.rule_name = pipeline.Config(self.ws.config_path).rule_name
        self.assertNotIn("평가 규칙:", text)

    def test_run_meta(self):
        meta = pipeline.read_json(self.ws.config.results_dir / "ft_seed52" / "run_meta.json")
        self.assertEqual((meta["seed"], meta["hash_scope"], meta["model_id"]), (52, "both", "mock"))
        self.assertEqual(len(meta["dataset_hash"]), 64)
        self.assertEqual(len(meta["prompt_hash"]), 64)
        data = self.ws.data
        self.assertEqual(meta["dataset_hash"], data.dataset_hash("both"))
        self.assertNotEqual(data.dataset_hash("both"), data.dataset_hash("A"))
        self.assertNotEqual(data.dataset_hash("A"), data.dataset_hash("B"))

    def test_comparison(self):
        a = self.comparison["a"]
        self.assertEqual(a["mode"], "compare")
        self.assertEqual(a["n_seeds"], 5)
        self.assertFalse(a["reference_only"])
        self.assertEqual(a["missing_seeds"], [])
        self.assertEqual(a["comparison"]["axis1"]["units"], 94)
        self.assertEqual(a["comparison"]["axis2"]["units"], 100)
        self.assertEqual(self.comparison["b"]["mode"], "compare")
        self.assertIsNotNone(self.comparison["b"]["fm"]["ft_sd"])

    def test_report_content(self):
        text = self.text
        self.assertTrue(text.startswith("| 축 | 지표 |"))        # 표로 시작한다
        name = self.ws.config.rule_name
        self.assertTrue(name)                                      # 설정에 규칙 이름이 있다
        self.assertEqual(text.count(name), 1)                      # 규칙 이름은 한 번만, 맨 끝 기록에 남긴다
        self.assertGreater(text.index(name), text.index("### 실행 기록"))
        for needle in ("| 1 | 값 정확도 (주) |", "| 2 | F1-OOS |",
                       "| 3 | FM |", "| 3 | AM |", "코퍼스 규모 미달", "ref_coverage", "참조 1개", "종합 판정",
                       "보정하지 않은 95% 구간", "사례 유형별 값 정확도", "표본 부족", "dataset_hash"):
            self.assertIn(needle, text)
        self.assertNotIn("참고용 (시드 N<5)", text.split("## 기록")[0])


class Modes(unittest.TestCase):
    def test_base_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            ws.run("base", 0.8, seed=1)
            comparison, text = ws.report_text()
        self.assertEqual(comparison["a"]["mode"], "base_only")
        self.assertEqual(comparison["a"]["overall"], "미판정")
        self.assertIn("— (출력 1벌)", text)
        self.assertIn("**종합 판정: 미판정**", text)

    def test_ft_only_with_fewer_than_five_and_one_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            ws.run("ft_seed42", 0.9, seed=2)
            comparison, text = ws.report_text()
        self.assertEqual(comparison["a"]["mode"], "ft_only")
        self.assertEqual(comparison["a"]["missing_seeds"], [52, 62, 72, 82])
        self.assertEqual(comparison["b"]["fm"]["ft_sd"], None)
        self.assertIn("표준편차 없음", text)
        self.assertIn("— (base 없음)", text)
        self.assertIn("**종합 판정: 미판정**", text)

    def test_compare_with_three_seeds_is_marked_reference_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            ws.run("base", 0.5, seed=1)
            for i, seed in enumerate((42, 52, 62)):
                ws.run("ft_seed%d" % seed, 0.9, seed=20 + i)
            comparison, text = ws.report_text()
        a = comparison["a"]
        self.assertEqual((a["mode"], a["n_seeds"], a["reference_only"]), ("compare", 3, True))
        self.assertIn("참고용 (시드 N<5)", text)
        self.assertIsNotNone(a["comparison"]["axis1"]["ci_adjusted"])

    def test_only_role_a_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            ws.run("base", 0.8, seed=1, roles=("A",))
            ws.run("ft_seed42", 0.9, seed=2, roles=("A",))
            meta = pipeline.read_json(ws.config.results_dir / "base" / "run_meta.json")
            comparison, text = ws.report_text()
            self.assertFalse((ws.config.results_dir / "base" / "m2p_scores.jsonl").exists())
        self.assertEqual(meta["hash_scope"], "A")
        self.assertEqual(comparison["b"]["mode"], "none")
        self.assertIn("| 3 | FM | — | — | — | — |", text)

    def test_missing_outputs_are_warned_and_scored_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            folder = write_mock_run(ws.config.runs_dir, "base", ws.data, 1.0, 0)
            lines = (folder / "m1v_outputs.jsonl").read_text(encoding="utf-8").splitlines()
            (folder / "m1v_outputs.jsonl").write_text("\n".join(lines[:90]) + "\n", encoding="utf-8")
            info = pipeline.score_run_folder(ws.config, ws.data, "base")
            summary = pipeline.read_json(ws.config.results_dir / "base" / "m1v_summary.json")
        self.assertEqual(summary["diagnostics"]["missing_outputs"], 10)
        self.assertTrue(any("10 of 100" in w for w in info["warnings"]))

    def test_unknown_run_folder_and_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            with self.assertRaises(InputError):
                pipeline.score_run_folder(ws.config, ws.data, "base")
            (ws.config.runs_dir / "base").mkdir(parents=True)
            with self.assertRaises(InputError):
                pipeline.score_run_folder(ws.config, ws.data, "base")

    def test_external_corpus_is_used_and_references_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            sentences = [ws.data.reference(ws.data.ids[0])] + ["안녕하세요 반갑습니다", "오늘 날씨가 좋네요 정말"]
            ws.config.corpus_path.parent.mkdir(parents=True, exist_ok=True)
            ws.config.corpus_path.write_text("\n".join(sentences) + "\n", encoding="utf-8")
            info = ws.run("base", 1.0, seed=0, roles=("B",))
            b = pipeline.read_json(ws.config.results_dir / "base" / "m2p_summary.json")
        self.assertEqual(b["corpus_sentences"], 2)
        self.assertTrue(any("left out" in w for w in info["warnings"]))


SETTINGS = {"temperature": 0, "max_new_tokens_m1v": 1024, "max_new_tokens_m2p": 128, "stop": [],
            "json_mode_m1v": False}


class DecodingSettings(unittest.TestCase):
    """디코딩 설정은 정하지 않고 받은 값을 그대로 기록하고, 실행 사이에 같은지만 확인한다."""

    def test_recorded_as_given_and_warned_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            given = ws.run("base", 0.8, seed=1, decoding=SETTINGS)
            none = ws.run("ft_seed42", 0.9, seed=2)
            meta_given = pipeline.read_json(ws.config.results_dir / "base" / "run_meta.json")
            meta_none = pipeline.read_json(ws.config.results_dir / "ft_seed42" / "run_meta.json")
        self.assertEqual(meta_given["decoding"], SETTINGS)
        self.assertFalse(any("decoding" in w for w in given["warnings"]))
        self.assertIsNone(meta_none["decoding"])
        self.assertTrue(any("decoding settings were not provided" in w for w in none["warnings"]))

    def test_our_config_does_not_dictate_the_settings(self):
        self.assertFalse(hasattr(config(), "decoding"))
        self.assertNotIn("decoding", config().raw)

    def test_non_object_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            write_mock_run(ws.config.runs_dir, "base", ws.data, 1.0, 0)
            with self.assertRaises(InputError):
                pipeline.score_run_folder(ws.config, ws.data, "base", decoding=["temperature", 0])

    def compare_with(self, base, *fts):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            ws.run("base", 0.6, seed=1, decoding=base)
            for i, (seed, settings) in enumerate(zip((42, 52, 62, 72, 82), fts)):
                ws.run("ft_seed%d" % seed, 0.9, seed=10 + i, decoding=settings)
            comparison, text = ws.report_text()
        return comparison["decoding"], text

    def test_same_settings_everywhere(self):
        decoding, text = self.compare_with(SETTINGS, SETTINGS, dict(reversed(list(SETTINGS.items()))))
        self.assertTrue(decoding["consistent"])
        self.assertEqual((decoding["differences"], decoding["missing"]), ([], []))
        self.assertIn("설정이 제공된 모든 실행의 값이 같다", text)
        self.assertNotIn("경고: 실행 사이에 디코딩 설정이 다르다", text)

    def test_different_settings_are_flagged_but_scores_are_untouched(self):
        other = dict(SETTINGS, max_new_tokens_m1v=512, json_mode_m1v=True)
        decoding, text = self.compare_with(SETTINGS, other, SETTINGS)
        self.assertFalse(decoding["consistent"])
        self.assertEqual([d["key"] for d in decoding["differences"]], ["json_mode_m1v", "max_new_tokens_m1v"])
        self.assertEqual(decoding["differences"][1]["values"], {"base": 1024, "ft_seed42": 512, "ft_seed52": 1024})
        self.assertIn("경고: 실행 사이에 디코딩 설정이 다르다", text)
        self.assertIn("`max_new_tokens_m1v`: base=1024, ft_seed42=512, ft_seed52=1024", text)
        self.assertIn("JSON 모드가 켜진 실행이 있다", text)
        self.assertIn("| 1 | 값 정확도 (주) |", text)                     # 점수와 판정은 그대로 나온다

    def test_missing_settings_and_too_few_to_compare(self):
        decoding, text = self.compare_with(SETTINGS, None)
        self.assertIsNone(decoding["consistent"])
        self.assertEqual(decoding["missing"], ["ft_seed42"])
        self.assertIn("설정이 제공된 실행이 둘 미만", text)
        self.assertIn("설정이 제공되지 않은 실행: ft_seed42", text)
        self.assertIn("- `ft_seed42`: 미제공", text)

    def test_key_present_in_only_some_runs_counts_as_a_difference(self):
        smaller = {k: v for k, v in SETTINGS.items() if k != "json_mode_m1v"}
        decoding, _ = self.compare_with(SETTINGS, smaller)
        self.assertFalse(decoding["consistent"])
        self.assertEqual(decoding["differences"][0]["values"], {"base": False, "ft_seed42": "(없음)"})

    def test_no_settings_at_all(self):
        decoding, text = self.compare_with(None, None)
        self.assertIsNone(decoding["consistent"])
        self.assertEqual(sorted(decoding["missing"]), ["base", "ft_seed42"])
        self.assertIn("- `base`: 미제공", text)


class Cli(unittest.TestCase):
    def test_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            args = ["--config", str(ws.config_path)]
            write_mock_run(ws.config.runs_dir, "base", ws.data, 0.7, 1)
            write_mock_run(ws.config.runs_dir, "ft_seed42", ws.data, 0.9, 2)
            outputs = {}
            for command in (["validate"], ["build-corpus"], ["score", "base", "--model-id", "m"],
                            ["score", "ft_seed42"], ["compare"], ["report"]):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    code = cli.main(args + command)
                outputs[command[0]] = (code, buffer.getvalue())
                self.assertEqual(code, 0, outputs[command[0]])
            self.assertTrue((ws.config.results_dir / "report.md").exists())
            self.assertIn("32", outputs["build-corpus"][1])
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli.main(args + ["score", "nope"])
            self.assertEqual(code, 2)

    def test_run_id_is_positional_and_the_old_flag_is_gone(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            write_mock_run(ws.config.runs_dir, "base", ws.data, 0.9, 1)
            args = ["--config", str(ws.config_path), "score"]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(args + ["base"]), 0)
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    cli.main(args + ["--run", "base"])             # 옛 형식은 받아들이지 않는다
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    cli.main(args)                                  # 실행 이름이 없으면 사용법 오류

    def test_decoding_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(tmp)
            write_mock_run(ws.config.runs_dir, "base", ws.data, 0.9, 1)
            good = Path(tmp) / "settings.json"
            good.write_text(json.dumps(SETTINGS), encoding="utf-8")
            bad = Path(tmp) / "bad.json"
            bad.write_text("not json", encoding="utf-8")
            listed = Path(tmp) / "list.json"
            listed.write_text("[1, 2]", encoding="utf-8")
            args = ["--config", str(ws.config_path), "score", "base"]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(args + ["--decoding", str(good)]), 0)
                self.assertEqual(cli.main(args + ["--decoding", str(bad)]), 2)
                self.assertEqual(cli.main(args + ["--decoding", str(Path(tmp) / "missing.json")]), 2)
                self.assertEqual(cli.main(args + ["--decoding", str(listed)]), 2)
            meta = pipeline.read_json(ws.config.results_dir / "base" / "run_meta.json")
        self.assertEqual(meta["decoding"], SETTINGS)


if __name__ == "__main__":
    unittest.main()
