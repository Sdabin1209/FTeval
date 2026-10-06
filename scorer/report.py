"""report.md 생성."""
import json

from . import pipeline
from .pipeline import BASE, DiskStore

def fmt(value, digits=4):
    """숫자를 소수 digits자리 문자열로 바꾼다. None이면 '—'."""
    return "—" if value is None else ("%.*f" % (digits, value))


def fmt_pair(low_high):
    """[하한, 상한] 구간을 문자열로 바꾼다."""
    return "[%s, %s]" % (fmt(low_high[0]), fmt(low_high[1]))


def _verdict_text(metric, reference_only, n_seeds):
    """표의 판정 열 문구를 만든다. '개선 / 통과'처럼 판정과 합격을 함께 적고, 표본 부족 같은 상태가 있으면 그것을 적고, 시드가 5벌 미만이면
    '참고용'을 덧붙인다.
    """
    status = metric["status"]
    text = status if status else "%s / %s" % (metric["verdict"], metric["pass"])
    if status:
        text += " / %s" % metric["pass"]
    if reference_only:
        text += " (참고용 (시드 N<5), N=%d)" % n_seeds
    return text


def _a_rows(a, label_ft="ft (시드 평균)"):
    """축 1·2의 표 행. 비교 / base만 / ft만 세 경우를 나눈다."""
    mode = a["mode"]
    rows = []
    if mode == "none":
        return [("1", "값 정확도 (주)", "—", "—", "—", "— (출력 없음)"),
                ("2", "F1-OOS", "—", "—", "—", "— (출력 없음)")]
    ft = a["ft_mean"] or {}
    base = a["base"] or {}
    for axis, name, key in (("1", "값 정확도 (주)", "value_accuracy"), ("2", "F1-OOS", "f1_oos")):
        if mode == "compare":
            metric = a["comparison"]["axis1" if axis == "1" else "axis2"]
            interval = fmt_pair(metric["ci_adjusted"]) if metric["ci_adjusted"] else "—"
            rows.append((axis, name, fmt(ft.get(key)), fmt(base.get(key)), interval,
                         _verdict_text(metric, a["reference_only"], a["n_seeds"])))
        elif mode == "base_only":
            rows.append((axis, name, "— (ft 없음)", fmt(base.get(key)), "— (출력 1벌)", "— (출력 1벌)"))
        else:
            rows.append((axis, name, fmt(ft.get(key)), "— (base 없음)", "— (base 없음)", "— (base 없음)"))
    aux = ("1", "스키마 정확도 / 핵심 일치율 / 칸 단위 micro F1 (보조)",
           " / ".join(fmt(ft.get(k)) for k in ("schema_accuracy", "core_match_rate", "fact_micro_f1")) if ft else "—",
           " / ".join(fmt(base.get(k)) for k in ("schema_accuracy", "core_match_rate", "fact_micro_f1")) if base else "—",
           "—", "—")
    rows.insert(1, aux)
    return rows


def _b_rows(b):
    """표의 FM·AM 행. 판정은 하지 않으므로 판정 열은 '— (판정하지 않음)'."""
    rows = []
    for axis_label, key in (("FM", "fm"), ("AM", "am")):
        entry = b.get(key) if b["mode"] != "none" else None
        if not entry:
            rows.append(("3", axis_label, "—", "—", "—", "—"))
            continue
        sd = entry.get("ft_sd")
        if "ft_mean" in entry:
            sd_text = "표준편차 %s" % fmt(sd) if sd is not None else "표준편차 없음"
        else:
            sd_text = "—"
        rows.append(("3", axis_label,
                     fmt(entry.get("ft_mean")) if "ft_mean" in entry else "— (ft 없음)",
                     fmt(entry.get("base_mean")) if "base_mean" in entry else "— (base 없음)",
                     sd_text, "— (판정하지 않음)"))
    return rows


def _dist_line(name, d):
    """분포(최소, 25·50·75%, 최대)를 한 줄 문장으로 바꾼다."""
    if not d:
        return "%s: —" % name
    return "%s: 최소 %s, 25%% %s, 50%% %s, 75%% %s, 최대 %s" % (
        name, fmt(d["min"]), fmt(d["p25"]), fmt(d["p50"]), fmt(d["p75"]), fmt(d["max"]))


def build_report(config, comparison, data, store=None):
    """report.md 전체 글을 만든다. 요약 표, 종합 판정, 기록(구간, 진단, 사례 유형별, B 점수, 실행 기록) 순서."""
    store = store if store is not None else DiskStore(config.results_dir)
    a, b = comparison["a"], comparison["b"]
    lines = ["| 축 | 지표 | ft (시드 평균) | base | 보정 구간 | 판정 |", "|---|---|---|---|---|---|"]
    for row in _a_rows(a) + _b_rows(b):
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    overall = a["overall"]
    lines.append("**종합 판정: %s**" % overall)
    if a["mode"] == "compare" and a["reference_only"]:
        lines.append("(참고용 (시드 N<5), N=%d)" % a["n_seeds"])
    elif a["mode"] in ("base_only", "ft_only"):
        lines.append("(비교할 수 없어 미판정: %s)" % ("출력 1벌" if a["mode"] == "base_only" else "base 없음"))
    lines.append("")

    lines.append("## 기록")
    lines.append("")
    # 시드
    for name, part in (("A (값 뽑기)", a), ("B (문장 만들기)", b)):
        runs = ", ".join(part["ft_runs"]) or "없음"
        lines.append("- %s: 모드 `%s`, ft 실행 %d개(%s), 설정 시드 중 폴더 없는 시드: %s%s" % (
            name, part["mode"], part["n_seeds"], runs,
            ", ".join(map(str, part["missing_seeds"])) or "없음",
            ", 설정에 없는 시드: %s" % ", ".join(map(str, part["extra_seeds"])) if part["extra_seeds"] else ""))
    lines.append("- 설정 시드 목록: %s" % ", ".join(map(str, config.seeds)))
    lines.append("")

    # 축 1·2 기록
    if a["mode"] == "compare":
        lines.append("### 보정하지 않은 95% 구간 (참고)")
        for key in ("axis1", "axis2"):
            metric = a["comparison"][key]
            lines.append("- %s: %s (단위 수 %d, 임계값 %s)" % (
                metric["name"], fmt_pair(metric["ci_95"]) if metric["ci_95"] else (metric["status"] or "—"),
                metric["units"], fmt(metric["threshold"], 2)))
        boot = a["comparison"]["bootstrap"]
        lines.append("- 부트스트랩: B=%d, alpha=%s, 보정 alpha=%s, 난수 시드 %d, ft 시드 실행 %d개" % (
            boot["B"], boot["alpha"], boot["adjusted_alpha"], boot["rng_seed"], boot["seed_runs"]))
        lines.append("")
    run_for_detail = BASE if a["base"] is not None else (a["ft_runs"][0] if a["ft_runs"] else None)
    if run_for_detail:
        summary = store.read_json(run_for_detail, "m1v_summary.json")
        diag = summary["diagnostics"]
        lines.append("### A 진단 (`%s`)" % run_for_detail)
        lines.append("- 파싱 실패 %d, 스키마 실패 %d, 값 규칙 위반 %d, 정답이 값인데 규격 외로 판정 %d, 출력 결측 %d" % (
            diag["parse_failures"], diag["schema_failures"], diag["values_violations"],
            diag["value_cases_judged_out_of_scope"], diag["missing_outputs"]))
        lines.append("- 상태별 재현율: " + ", ".join(
            "%s %s (정답 %d칸)" % (k, fmt(v), summary["status_gold_counts"][k])
            for k, v in summary["status_recall"].items()))
        lines.append("- 의도별 재현율: " + ", ".join(
            "%s %s (%d건)" % (k, fmt(v), summary["intent_gold_counts"][k])
            for k, v in summary["intent_recall"].items()))
        lines.append("")
        lines.append("### 사례 유형별 값 정확도 (`%s`, 점 추정치)" % run_for_detail)
        for name, entry in summary["by_category"].items():
            note = " — 표본 부족" if entry["small_sample"] else ""
            lines.append("- %s: %s (%d건)%s" % (name, fmt(entry["value_accuracy"]), entry["n"], note))
        lines.append("")
    # B 기록
    run_for_b = BASE if "base_mean" in b.get("fm", {}) else (b["ft_runs"][0] if b["ft_runs"] else None)
    if run_for_b:
        summary = store.read_json(run_for_b, "m2p_summary.json")
        lines.append("### B 문장 점수 기록")
        scorer_info = summary.get("scorer") or {"type": "ngram"}
        if scorer_info["type"] == "neural":
            lm, emb = scorer_info["lm"], scorer_info["embedder"]
            lines.append("- 문장 점수 방식: neural (코퍼스 없음). FM 모델 `%s`(revision %s), AM 모델 `%s`(revision %s, 풀링 %s), 장치 %s" % (
                lm["model"], lm["revision"] or "미지정", emb["model"], emb["revision"] or "미지정",
                emb["pooling"], scorer_info["device"]))
        else:
            if summary["corpus_below_minimum"]:
                lines.append("- 코퍼스 규모 미달: %d문장 (요구 %d문장). 점수의 절대값은 믿기 어렵다." % (
                    summary["corpus_sentences"], summary["corpus_minimum"]))
            else:
                lines.append("- 코퍼스 %d문장" % summary["corpus_sentences"])
            lines.append("- ref_coverage(정답 문장 어절 중 코퍼스에 있는 비율): %s" % fmt(summary["ref_coverage"]))
        lines.append("- 참조 1개 (사례당 정답 문장 1개, 서로 다른 문장 %d개)" % summary["distinct_references"])
        if scorer_info["type"] == "ngram":
            lines.append("- 후보의 미등록 어절 비율 평균 %s, 미등록 bigram 비율 평균 %s (`%s`)" % (
                fmt(summary["oov_word_mean"]), fmt(summary["oov_bigram_mean"]), run_for_b))
        if comparison.get("sentence_scorer", {}).get("consistent") is False:
            lines.append("- **경고: 실행마다 문장 점수 방식이 다릅니다. FM·AM 점수끼리 비교할 수 없습니다.**")
        for label, key in (("FM", "fm"), ("AM", "am")):
            entry = b[key]
            if "base_distribution" in entry:
                lines.append("- " + _dist_line("%s 사례별 분포 (base)" % label, entry["base_distribution"]))
            if "ft_distribution" in entry:
                lines.append("- " + _dist_line("%s 사례별 분포 (ft, 사례마다 시드 평균)" % label,
                                              entry["ft_distribution"]))
        lines.append("")
    # 실행 목록: base가 있으면 먼저, 그 뒤에 ft 실행을 시드 순서로
    all_runs = ([BASE] if a["base"] is not None or "base_mean" in b.get("fm", {}) else []) + sorted(
        set(a["ft_runs"]) | set(b["ft_runs"]))
    # 근거 유효율과 계획 위반 표현 (실행별 보조 지표)
    extra = []
    for run in all_runs:
        parts = []
        if store.has(run, "m1v_summary.json"):
            ev = store.read_json(run, "m1v_summary.json").get("evidence")
            if ev:
                parts.append("A 근거 유효 출력 %s (%d/%d), 근거 항목 유효 %s (%d/%d)" % (
                    fmt(ev["output_valid_rate"]), ev["outputs_valid"], ev["outputs_checked"],
                    fmt(ev["item_valid_rate"]), ev["valid_items"], ev["items"]))
        if store.has(run, "m2p_summary.json"):
            bs = store.read_json(run, "m2p_summary.json")
            if "completion_claims" in bs:
                parts.append("B 계획 위반 표현 %d건 (결측 제외 비율 %s)" % (
                    bs["completion_claims"], fmt(bs["completion_claim_rate"])))
        if parts:
            extra.append("- `%s`: %s" % (run, ", ".join(parts)))
    if extra:
        lines.append("### 근거 유효율과 계획 위반 표현 (실행별, 판정에 쓰지 않음)")
        lines += extra
        lines.append("")
    # 디코딩 설정: 팀이 알려 준 값을 그대로 보여 주고, 실행 사이에 같은지만 알린다
    decoding = comparison.get("decoding")
    if decoding and decoding["per_run"]:
        lines.append("### 디코딩 설정 (모델을 돌린 쪽이 알려 준 값, 판정에 쓰지 않음)")
        for run, settings in decoding["per_run"].items():
            lines.append("- `%s`: %s" % (run, json.dumps(settings, ensure_ascii=False) if settings is not None
                                         else "미제공"))
        if decoding["consistent"] is True:
            lines.append("- 설정이 제공된 모든 실행의 값이 같다.")
        elif decoding["consistent"] is False:
            lines.append("- **경고: 실행 사이에 디코딩 설정이 다르다.** base와 ft의 점수 차이에 모델 차이와 설정 차이가 "
                         "섞여 있을 수 있다. 점수와 판정은 그대로 계산했다.")
            for diff in decoding["differences"]:
                lines.append("  - `%s`: %s" % (diff["key"], ", ".join(
                    "%s=%s" % (run, json.dumps(v, ensure_ascii=False)) for run, v in diff["values"].items())))
        else:
            lines.append("- 설정이 제공된 실행이 둘 미만이라 서로 비교하지 못했다.")
        if decoding["missing"]:
            lines.append("- 설정이 제공되지 않은 실행: %s" % ", ".join(decoding["missing"]))
        if any((s or {}).get("json_mode_m1v") for s in decoding["per_run"].values()):
            lines.append("- A 호출에 JSON 모드가 켜진 실행이 있다. 이 경우 스키마 정확도는 모델이 스스로 형식을 "
                         "지킨 정도가 아닐 수 있다.")
        lines.append("")
    # 해시와 실행 기록
    metas = []
    for run in all_runs:
        if store.has(run, "run_meta.json"):
            metas.append((run, store.read_json(run, "run_meta.json")))
    lines.append("### 실행 기록")
    for run, meta in metas:
        lines.append("- `%s`: dataset_hash `%s`, prompt_hash `%s`, hash_scope `%s`, 모델 %s" % (
            run, meta["dataset_hash"][:16] + "…", meta["prompt_hash"][:16] + "…", meta["hash_scope"],
            meta["model_id"] or "(미지정)"))
        for warning in meta.get("warnings", []):
            lines.append("  - 경고: %s" % warning)
    if config.rule_name:        # 어느 규칙으로 돌렸는지 가볍게 남기는 표시. 이름은 설정에서 읽는다
        lines.append("- 평가 규칙: %s" % config.rule_name)
    lines.append("- 설정: B=%d, alpha=%s, rng_seed=%d, min_units=%d, 임계값 %s, 코퍼스 %s" % (
        config.ci["B"], config.ci["alpha"], config.ci["rng_seed"], config.ci["min_units"],
        config.thresholds, config.corpus_path.name if config.sentence_scorer.get("type") == "ngram" else "(neural: 없음)"))
    lines.append("")
    return "\n".join(lines)


def write_report(config, comparison, data, store=None):
    """build_report로 만든 글을 results/report.md에 쓰고 그 경로를 돌려준다."""
    text = build_report(config, comparison, data, store)
    path = config.results_dir / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


__all__ = ["build_report", "write_report", "pipeline"]
