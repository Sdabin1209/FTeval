"""역할 A(값 뽑기) 채점."""
import json
from collections import Counter

from . import checks, schema
from .metrics import f1_from_counts, prf
from .strictjson import StrictJSONError, parse_strict_object

STATUSES_WITH_RECALL = ("MISSING", "AMBIGUOUS", "UNCERTAIN")


def norm_value(value):
    """JSON 값의 해시 가능한 형태. 4와 4.0은 같고 True와 1은 다르다."""
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("num", value)
    if isinstance(value, str):
        return ("str", value)
    return ("json", json.dumps(value, sort_keys=True, ensure_ascii=False))


def comparison_sets(obj):
    """정확 일치 판정에 쓰는 집합들. obj는 스키마 검사를 통과한 것이어야 한다."""
    facts = set()
    core = set()
    for f in obj["facts"]:
        facts.add((f["item_id"], f["field"], norm_value(f["value"]), f["semantic_status"], f["precision"]))
        core.add((f["item_id"], f["field"], norm_value(f["value"])))
    intents = {(i["item_id"], i["intent"]) for i in obj["intents"]}
    relations = set()
    for r in obj["relations"]:
        t = r["correction_target"] or {}
        relations.add((r["item_id"], r["relation"], t.get("item_id"), t.get("field"), t.get("turn_index")))
    unmapped = Counter((u["scope"], u["evidence"]["text"]) for u in obj["unmapped_facts"])
    return {"facts": facts, "core": core, "intents": intents, "relations": relations, "unmapped": unmapped}


def is_out_of_scope(obj):
    """ANSWER가 아닌 의도가 하나라도 있으면 규격 외이다."""
    return any(i["intent"] != "ANSWER" for i in obj["intents"])


def prepare_gold(gold_obj):
    """정답 JSON을 채점용으로 준비한다. 비교용 집합, 정답이 규격 외인지, 원본 객체를 담는다."""
    sets = comparison_sets(gold_obj)
    return {"obj": gold_obj, "sets": sets, "oos": is_out_of_scope(gold_obj)}


def score_case(raw_output, gold, questionnaire_ids, context):
    """사례 하나를 채점한다. 출력이 없으면 raw_output은 None이거나 문자열이 아니다.

    gold는 prepare_gold의 결과이고, context는 입력 대화(근거 검사에 쓴다). 점수 필드와 함께,
    요약에 쓸 읽은 객체(pred_obj, 스키마를 통과한 경우에만)를 담은 dict를 돌려준다.
    """
    gold_oos = gold["oos"]
    result = {"missing": False, "parse_ok": False, "schema_ok": False, "values_ok": None,
              "correct": False, "core_correct": False, "pred_oos": not gold_oos,
              "evidence_ok": None, "evidence_items": 0, "evidence_valid_items": 0,
              "pred_obj": None, "errors": []}
    if not isinstance(raw_output, str):
        result["missing"] = True
        result["errors"].append("output missing")
        return result
    try:
        obj = parse_strict_object(raw_output)
    except StrictJSONError as exc:
        result["errors"].append("parse: %s" % exc)
        return result
    result["parse_ok"] = True
    problems = schema.schema_errors(obj)
    if problems:
        result["errors"] += problems
        return result
    result["schema_ok"] = True
    result["pred_obj"] = obj
    value_problems = schema.value_errors(obj, questionnaire_ids)
    result["values_ok"] = not value_problems
    result["errors"] += value_problems
    result["pred_oos"] = is_out_of_scope(obj)
    # 근거 검사는 값 규칙과 별개의 보조 지표라서 정오(correct)에는 쓰지 않는다.
    items, valid, all_valid = checks.evidence_report(obj, context)
    result["evidence_items"], result["evidence_valid_items"], result["evidence_ok"] = items, valid, all_valid
    if result["values_ok"]:
        pred = comparison_sets(obj)
        gset = gold["sets"]
        rest = (pred["intents"] == gset["intents"] and pred["relations"] == gset["relations"]
                and pred["unmapped"] == gset["unmapped"])
        result["correct"] = rest and pred["facts"] == gset["facts"]
        result["core_correct"] = rest and pred["core"] == gset["core"]
    return result


SCORE_KEYS = ("id", "missing", "parse_ok", "schema_ok", "values_ok", "correct", "core_correct",
              "evidence_ok", "gold_oos", "pred_oos", "category")


def score_row(case_id, category, gold, result):
    """score_case 결과를 점수 파일(m1v_scores.jsonl) 한 줄의 모양으로 정리한다."""
    row = {"id": case_id, "missing": result["missing"], "parse_ok": result["parse_ok"],
           "schema_ok": result["schema_ok"], "values_ok": result["values_ok"],
           "correct": result["correct"], "core_correct": result["core_correct"],
           "evidence_ok": result["evidence_ok"], "gold_oos": gold["oos"],
           "pred_oos": result["pred_oos"], "category": category}
    assert tuple(row) == SCORE_KEYS
    return row


def summarize(rows, results, golds, min_units):
    """실행 하나의 주 지표와 보조 지표 (m1v_summary.json에 저장되는 내용).

    rows, results, golds는 나란한 목록이다(점수 행, score_case 결과, prepare_gold 결과).
    """
    n = len(rows)
    in_scope = [i for i in range(n) if not golds[i]["oos"]]
    accuracy = (sum(rows[i]["correct"] for i in in_scope) / len(in_scope)) if in_scope else None
    tp = sum(1 for r in rows if r["pred_oos"] and r["gold_oos"])
    fp = sum(1 for r in rows if r["pred_oos"] and not r["gold_oos"])
    fn = sum(1 for r in rows if not r["pred_oos"] and r["gold_oos"])
    precision_oos = tp / (tp + fp) if tp + fp else 0.0
    recall_oos = tp / (tp + fn) if tp + fn else None

    passed = sum(1 for r in rows if r["schema_ok"] and r["values_ok"])
    core = (sum(rows[i]["core_correct"] for i in in_scope) / len(in_scope)) if in_scope else None

    # (item_id, field, value, status, precision) 일치에 대한 micro P/R/F1
    hit = pred_total = gold_total = 0
    status_hit = Counter()
    status_total = Counter()
    for res, gold in zip(results, golds):
        gold_facts = gold["sets"]["facts"]
        gold_total += len(gold_facts)
        if res["pred_obj"] is not None:
            pred_facts = comparison_sets(res["pred_obj"])["facts"]
            pred_total += len(pred_facts)
            hit += len(pred_facts & gold_facts)
            pred_status = {(f[0], f[1]): f[3] for f in pred_facts}
        else:
            pred_status = {}
        for f in gold_facts:
            if f[3] in STATUSES_WITH_RECALL:
                status_total[f[3]] += 1
                if pred_status.get((f[0], f[1])) == f[3]:
                    status_hit[f[3]] += 1
    micro_p, micro_r, micro_f1 = prf(hit, pred_total, gold_total)

    intent_hit = Counter()
    intent_total = Counter()
    for res, gold in zip(results, golds):
        pred_intents = comparison_sets(res["pred_obj"])["intents"] if res["pred_obj"] is not None else set()
        for item_id, intent in gold["sets"]["intents"]:
            intent_total[intent] += 1
            if (item_id, intent) in pred_intents:
                intent_hit[intent] += 1

    missing = sum(1 for r in rows if r["missing"])
    parse_fail = sum(1 for r in rows if not r["missing"] and not r["parse_ok"])
    schema_fail = sum(1 for r in rows if r["parse_ok"] and not r["schema_ok"])
    values_violation = sum(1 for r in rows if r["schema_ok"] and not r["values_ok"])
    false_oos = sum(1 for r in rows if r["schema_ok"] and r["pred_oos"] and not r["gold_oos"])

    checked = [res for res in results if res["evidence_ok"] is not None]
    evidence_items = sum(res["evidence_items"] for res in checked)
    evidence_valid_items = sum(res["evidence_valid_items"] for res in checked)
    evidence = {
        "outputs_checked": len(checked),
        "outputs_valid": sum(1 for res in checked if res["evidence_ok"]),
        "output_valid_rate": (sum(1 for res in checked if res["evidence_ok"]) / len(checked)) if checked else None,
        "items": evidence_items,
        "valid_items": evidence_valid_items,
        "item_valid_rate": (evidence_valid_items / evidence_items) if evidence_items else None,
    }

    by_category = {}
    for i in in_scope:
        entry = by_category.setdefault(rows[i]["category"], {"n": 0, "correct": 0})
        entry["n"] += 1
        entry["correct"] += int(rows[i]["correct"])
    categories = {}
    for name in sorted(by_category):
        entry = by_category[name]
        categories[name] = {"n": entry["n"], "value_accuracy": entry["correct"] / entry["n"],
                            "small_sample": entry["n"] < min_units}

    return {
        "n_cases": n,
        "n_in_scope": len(in_scope),
        "n_out_of_scope": n - len(in_scope),
        "value_accuracy": accuracy,
        "f1_oos": f1_from_counts(tp, fp, fn),
        "precision_oos": precision_oos,
        "recall_oos": recall_oos,
        "oos_counts": {"tp": tp, "fp": fp, "fn": fn},
        "schema_accuracy": passed / n if n else None,
        "core_match_rate": core,
        "fact_micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1,
                       "hit": hit, "pred_total": pred_total, "gold_total": gold_total},
        "status_recall": {s: (status_hit[s] / status_total[s] if status_total[s] else None)
                          for s in STATUSES_WITH_RECALL},
        "status_gold_counts": {s: status_total[s] for s in STATUSES_WITH_RECALL},
        "intent_recall": {k: intent_hit[k] / intent_total[k] for k in sorted(intent_total)},
        "intent_gold_counts": {k: intent_total[k] for k in sorted(intent_total)},
        "diagnostics": {"parse_failures": parse_fail, "schema_failures": schema_fail,
                        "values_violations": values_violation,
                        "value_cases_judged_out_of_scope": false_oos, "missing_outputs": missing},
        "evidence": evidence,
        "by_category": categories,
    }
