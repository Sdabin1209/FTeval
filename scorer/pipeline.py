"""실행 채점과 비교.

채점 계산(score_outputs)과 비교 계산(compare_runs)은 결과를 어디에 두는지(디스크 또는 메모리)와
무관하게 한 벌이다. 폴더에서 읽어 results/에 쓰는 함수(score_run_folder)와 메모리에서 바로
부르는 SDK(api.py)가 같은 계산을 쓴다.
"""
import json
import re
from pathlib import Path

import numpy as np

from . import scoring_a, scoring_b, stats
from .dataio import Dataset, InputError, read_outputs, validate_dataset
from .metrics import distribution, population_sd
from .store import DiskStore, MemoryStore, read_json, read_rows as _read_rows, write_json, write_jsonl

SCORER_VERSION = "1.0"
BASE = "base"
FT_PATTERN = re.compile(r"^ft_seed(\d+)$")

# 한 실행의 결과 파일 이름
A_SCORES, A_SUMMARY = "m1v_scores.jsonl", "m1v_summary.json"
B_SCORES, B_SUMMARY = "m2p_scores.jsonl", "m2p_summary.json"
RUN_META = "run_meta.json"

__all__ = ["Config", "DiskStore", "MemoryStore", "check_inputs", "load_corpus", "normalize_outputs",
           "score_outputs", "score_run_folder", "compare_runs", "read_json", "write_json", "write_jsonl"]


class Config:
    """config.json. 상대 경로는 설정 파일이 있는 폴더 기준이다."""

    def __init__(self, path):
        """config.json을 읽어 경로(데이터, runs, results, 코퍼스)와 CI, 임계값, 시드 같은 설정을 속성으로 펼친다."""
        self.path = Path(path).resolve()
        self.root = self.path.parent
        with open(self.path, encoding="utf-8") as fh:
            self.raw = json.load(fh)
        paths = self.raw["paths"]
        self.data_dir = self._resolve(paths["data_dir"])
        self.runs_dir = self._resolve(paths["runs_dir"])
        self.results_dir = self._resolve(paths["results_dir"])
        self.corpus_path = self._resolve(self.raw["lm_corpus"])
        self.ci = self.raw["ci"]
        self.thresholds = self.raw["thresholds"]
        self.seeds = self.raw["seeds"]
        self.corpus_min_sentences = self.raw.get("corpus_min_sentences", 18692)
        self.lsi_dim = self.raw.get("lsi_dim", 10)
        self.rule_name = self.raw.get("rule_name")        # 리포트 끝에 가볍게 남길 평가 규칙 이름(없어도 된다)

    def _resolve(self, value):
        """상대 경로를 설정 파일이 있는 폴더 기준 경로로 바꾼다."""
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)


def check_inputs(config):
    """평가 데이터 검증. (dataset, problems)를 돌려준다."""
    data = Dataset(config.data_dir)
    return data, validate_dataset(data)


def load_corpus(config, data, write=True):
    """코퍼스 문장. 코퍼스 파일이 아직 없으면 데이터에서 만든다.

    write가 True이면 만든 코퍼스를 파일로 쓰고(명령줄), False이면 파일을 만들지 않고 메모리에만
    둔다(SDK). 참조 문장은 코퍼스에 넣지 않는다. 데이터에서 만들 때는 build_corpus가 챗봇 응답
    문장 중 정답 문장이 아닌 것만 골라 두므로, 파일로 받은 코퍼스에서만 따로 뺀다.
    """
    path = config.corpus_path
    warnings = []
    if path.exists():
        lines = scoring_b.read_corpus(path)
        # 사람이 만든 코퍼스 파일에는 참조(정답) 문장이 섞여 있을 수 있어서 같은 줄을 뺀다
        references = set(data.references())
        kept = [line for line in lines if line not in references]
        if len(kept) != len(lines):
            warnings.append("%d corpus lines equal to a reference sentence were left out" % (len(lines) - len(kept)))
        lines = kept
    else:
        lines = scoring_b.build_corpus(data)       # 이미 참조 문장을 뺀 상태
        if write:
            scoring_b.write_corpus(lines, path)
            warnings.append("corpus file was missing and was generated at %s" % path)
        else:
            warnings.append("corpus file was missing and was built in memory from the data")
    return lines, warnings


def normalize_outputs(mapping, valid_ids, label):
    """메모리로 받은 모델 출력(id -> 값)을 파일에서 읽은 것과 같은 규칙으로 정리한다.

    평가 데이터에 없는 id나 문자열이 아닌 id는 무시하고 경고한다. 값이 문자열이 아닌 것은 그대로
    두며, 채점 때 결측으로 센다. (values, warnings)를 돌려준다.
    """
    if not isinstance(mapping, dict):
        raise InputError("%s must be a dict that maps case id to the model output" % label)
    values, warnings = {}, []
    for key, value in mapping.items():
        if not isinstance(key, str) or key not in valid_ids:
            warnings.append("%s: id %s is not an evaluation case and was ignored"
                            % (label, json.dumps(key, ensure_ascii=False, default=str)))
            continue
        values[key] = value
    return values, warnings


def score_outputs(config, data, run_id, a_values, b_values, *, model_id=None, decoding=None,
                  sentence_scorer=None, input_warnings=()):
    """한 실행의 모델 출력을 채점해 (결과 파일 dict, 경고 목록)을 돌려준다. 파일을 쓰지 않는다.

    a_values, b_values는 id -> 모델 출력의 dict이고, 그 역할을 평가하지 않으면 None이다. 둘 중
    하나는 있어야 한다. 없는 id는 결측이다. B를 채점하려면 sentence_scorer가 필요하다.
    decoding은 모델을 돌린 쪽이 알려 준 디코딩 설정(dict)이다. 우리가 정하지 않고 받은 값을 그대로
    실행 기록에 남기며, 없으면 None으로 기록하고 경고를 싣는다. input_warnings는 출력을 읽을 때
    생긴 경고로, 실행 기록의 경고 앞에 붙는다.
    """
    if a_values is None and b_values is None:
        raise InputError("there is no model output to score for %s" % run_id)
    if decoding is not None and not isinstance(decoding, dict):
        raise InputError("decoding settings must be a JSON object")
    if b_values is not None and sentence_scorer is None:
        raise InputError("scoring role B needs a sentence scorer built from the corpus")
    warnings = list(input_warnings)
    if decoding is None:
        warnings.append("decoding settings were not provided")
    artifacts = {}
    min_units = config.ci["min_units"]

    if a_values is not None:
        rows, results, golds = [], [], []
        for case_id in data.ids:
            gold = data.gold_a(case_id)
            context = data.user_json(data.row_a(case_id))["context"]
            result = scoring_a.score_case(a_values.get(case_id), gold, data.questionnaire_ids(case_id), context)
            rows.append(scoring_a.score_row(case_id, data.category(case_id), gold, result))
            results.append(result)
            golds.append(gold)
        missing = sum(1 for r in rows if r["missing"])
        if missing:
            warnings.append("m1v_outputs: %d of %d cases have no usable output (scored as missing)"
                            % (missing, len(rows)))
        artifacts[A_SCORES] = rows
        artifacts[A_SUMMARY] = scoring_a.summarize(rows, results, golds, min_units)

    if b_values is not None:
        rows = scoring_b.score_run(data, b_values, sentence_scorer)
        missing = sum(1 for r in rows if r["missing"])
        if missing:
            warnings.append("m2p_outputs: %d of %d cases have no usable output (scored as 0)"
                            % (missing, len(rows)))
        artifacts[B_SCORES] = rows
        artifacts[B_SUMMARY] = scoring_b.summarize(rows, data, sentence_scorer, config.corpus_min_sentences)

    scope = "both" if a_values is not None and b_values is not None else ("A" if a_values is not None else "B")
    match = FT_PATTERN.match(run_id)
    artifacts[RUN_META] = {
        "run_id": run_id,
        "model_id": model_id,
        "seed": int(match.group(1)) if match else None,
        "decoding": decoding,
        "dataset_hash": data.dataset_hash(scope),
        "prompt_hash": data.prompt_hash(),
        "hash_scope": scope,
        "scorer_version": SCORER_VERSION,
        "finetune": None,
        "warnings": warnings,
    }
    return artifacts, warnings


def score_run_folder(config, data, run_id, model_id=None, decoding=None):
    """실행 폴더(runs/<run_id>/) 하나를 채점해 results/<run_id>/ 아래에 파일을 쓴다. 짧은 결과 dict를 돌려준다.

    출력 파일을 읽는 일과 결과 파일을 쓰는 일만 여기서 하고, 채점 계산은 score_outputs가 한다.
    """
    run_dir = config.runs_dir / run_id
    a_path = run_dir / "m1v_outputs.jsonl"
    b_path = run_dir / "m2p_outputs.jsonl"
    if not run_dir.is_dir():
        raise InputError("run folder %s does not exist" % run_dir)
    if not a_path.exists() and not b_path.exists():
        raise InputError("%s has neither m1v_outputs.jsonl nor m2p_outputs.jsonl" % run_dir)
    ids = set(data.ids)
    input_warnings = []
    a_values = b_values = scorer = None
    if a_path.exists():
        a_values, w = read_outputs(a_path, "raw_output", ids)
        input_warnings += w
    if b_path.exists():
        b_values, w = read_outputs(b_path, "candidate", ids)
        input_warnings += w
        corpus, w = load_corpus(config, data, write=True)
        input_warnings += w
        scorer = scoring_b.SentenceScorer(corpus, lsi_dim=config.lsi_dim)
    artifacts, warnings = score_outputs(config, data, run_id, a_values, b_values, model_id=model_id,
                                        decoding=decoding, sentence_scorer=scorer, input_warnings=input_warnings)
    produced = DiskStore(config.results_dir).write(run_id, artifacts)
    return {"run_id": run_id, "files": produced, "warnings": warnings}


def discover_runs(store, filename):
    """저장소에서 해당 점수 파일이 있는 실행 id: (base 유무, 정렬된 ft 실행 id 목록)."""
    base = store.has(BASE, filename)
    fts = []
    for run in store.runs():
        match = FT_PATTERN.match(run)
        if match and store.has(run, filename):
            fts.append((int(match.group(1)), run))
    return base, [name for _, name in sorted(fts)]


def _a_arrays(rows):
    """A 점수 행을 비교용 배열로 바꾼다. 값 사례의 correct(0/1), 전체의 pred_oos, 전체의 gold_oos."""
    correct = np.array([int(r["correct"]) for r in rows if not r["gold_oos"]], dtype=float)
    pred = np.array([r["pred_oos"] for r in rows], dtype=bool)
    gold = np.array([r["gold_oos"] for r in rows], dtype=bool)
    return correct, pred, gold


def _mean(values):
    """None을 뺀 평균. 값이 하나도 없으면 None."""
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def compare_runs(config, store=None):
    """저장소(기본은 results/ 폴더)의 base와 ft 실행을 비교해 비교 결과 dict를 돌려준다."""
    store = store if store is not None else DiskStore(config.results_dir)
    a, b = _compare_a(config, store), _compare_b(config, store)
    runs = ([BASE] if a["base"] is not None or "base_mean" in b.get("fm", {}) else []) + sorted(
        set(a["ft_runs"]) | set(b["ft_runs"]))
    return {"a": a, "b": b, "decoding": _decoding_check(runs, store)}


def _decoding_check(runs, store):
    """실행들의 디코딩 설정이 서로 같은지 확인한다. 설정을 정하지는 않고 받은 값만 비교한다.

    per_run: 실행별 설정(없으면 None), missing: 설정이 없는 실행, differences: 실행마다 다른 항목,
    consistent: 설정이 있는 실행이 둘 이상일 때 모두 같으면 True, 다르면 False, 비교할 수 없으면 None.
    """
    per_run = {}
    for run in runs:
        per_run[run] = store.read_json(run, RUN_META).get("decoding") if store.has(run, RUN_META) else None
    given = {run: d for run, d in per_run.items() if d is not None}
    differences = []
    for key in sorted({k for d in given.values() for k in d}):
        values = {run: d.get(key, "(없음)") for run, d in given.items()}
        if len({json.dumps(v, sort_keys=True, ensure_ascii=False) for v in values.values()}) > 1:
            differences.append({"key": key, "values": values})
    return {"per_run": per_run, "missing": [r for r in runs if per_run[r] is None],
            "differences": differences,
            "consistent": (not differences) if len(given) >= 2 else None}


def _seed_info(config, ft_runs):
    """ft 실행 목록으로 시드 수, 5벌 미만 여부(참고용), 설정에는 있는데 폴더가 없는 시드를 정리한다."""
    found = {int(FT_PATTERN.match(n).group(1)) for n in ft_runs}
    return {"ft_runs": ft_runs, "n_seeds": len(ft_runs), "reference_only": 0 < len(ft_runs) < 5,
            "missing_seeds": [s for s in config.seeds if s not in found],
            "extra_seeds": sorted(found - set(config.seeds))}


def _compare_a(config, store):
    """A(값 뽑기)의 base와 ft 실행을 모아 비교한다. 모드(compare, base_only, ft_only, none), 실행별 점수, ft 평균,
    부트스트랩 판정을 담은 dict를 돌려준다.
    """
    has_base, ft_runs = discover_runs(store, A_SCORES)
    info = _seed_info(config, ft_runs)
    info["mode"] = ("compare" if has_base and ft_runs else "base_only" if has_base
                    else "ft_only" if ft_runs else "none")
    info["overall"] = "미판정"
    info["comparison"] = None
    summaries = {}
    for run in ([BASE] if has_base else []) + ft_runs:
        summaries[run] = store.read_json(run, A_SUMMARY)
    info["per_run"] = {run: {
        "value_accuracy": s["value_accuracy"], "f1_oos": s["f1_oos"],
        "schema_accuracy": s["schema_accuracy"], "core_match_rate": s["core_match_rate"],
        "fact_micro_f1": s["fact_micro"]["f1"]} for run, s in summaries.items()}
    keys = ("value_accuracy", "f1_oos", "schema_accuracy", "core_match_rate", "fact_micro_f1")
    info["base"] = info["per_run"].get(BASE)
    info["ft_mean"] = ({k: _mean([info["per_run"][r][k] for r in ft_runs]) for k in keys}
                       if ft_runs else None)
    if info["mode"] == "compare":
        base_rows = store.read_rows(BASE, A_SCORES)
        base_correct, base_pred, gold = _a_arrays(base_rows)
        ft_correct, ft_pred = [], []
        for run in ft_runs:
            rows = store.read_rows(run, A_SCORES)
            if [r["id"] for r in rows] != [r["id"] for r in base_rows]:
                raise InputError("score files of %s and base do not list the same cases" % run)
            c, p, g = _a_arrays(rows)
            ft_correct.append(c)
            ft_pred.append(p)
        result = stats.compare(base_correct, np.array(ft_correct), base_pred, np.array(ft_pred),
                               gold, config.ci, config.thresholds)
        info["comparison"] = result
        info["overall"] = result["overall"]
    return info


def _compare_b(config, store):
    """B(문장 만들기)의 FM·AM을 실행별로 모은다. base 평균과 분포, ft 시드 평균과 시드 간 표준편차, 사례별 분포를 계산한다. 판정은 하지
    않는다.
    """
    has_base, ft_runs = discover_runs(store, B_SCORES)
    info = _seed_info(config, ft_runs)
    info["mode"] = ("compare" if has_base and ft_runs else "base_only" if has_base
                    else "ft_only" if ft_runs else "none")
    per_run = {}
    for run in ([BASE] if has_base else []) + ft_runs:
        rows = store.read_rows(run, B_SCORES)
        per_run[run] = {"ids": [r["id"] for r in rows], "FM": [r["FM"] for r in rows],
                        "AM": [r["AM"] for r in rows]}
    info["per_run"] = {run: {"FM": float(np.mean(v["FM"])), "AM": float(np.mean(v["AM"]))}
                       for run, v in per_run.items()}
    for metric in ("FM", "AM"):
        entry = {}
        if has_base:
            entry["base_mean"] = info["per_run"][BASE][metric]
            entry["base_distribution"] = distribution(per_run[BASE][metric])
        if ft_runs:
            means = [info["per_run"][r][metric] for r in ft_runs]
            entry["ft_mean"] = float(np.mean(means))
            entry["ft_sd"] = population_sd(means) if len(means) > 1 else None
            per_case = np.mean([per_run[r][metric] for r in ft_runs], axis=0)
            entry["ft_distribution"] = distribution(per_case)
        info[metric.lower()] = entry
    return info
