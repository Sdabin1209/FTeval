"""base와 파인튜닝 실행의 비교: 이중 부트스트랩과 판정."""
import numpy as np

from .metrics import f1_from_counts

SAMPLE_TOO_SMALL = "표본 부족"
RESAMPLING_FAILED = "재표집 실패"
UNDEFINED = "정의 불가"
IMPROVED, NO_DIFFERENCE, DEGRADED = "개선", "유의한 차이 없음", "저하"
PASS, FAIL, UNSETTLED = "통과", "미달", "불확정"
MAX_REDRAWS = 50


def f1_vector(pred, gold):
    """pred(S x n 불리언)의 행마다 gold(n 불리언)에 대한 F1-OOS. gold에 True가 하나 이상 있어야 한다."""
    pred = np.atleast_2d(pred)
    tp = (pred & gold).sum(axis=1).astype(float)
    fp = (pred & ~gold).sum(axis=1).astype(float)
    fn = (~pred & gold).sum(axis=1).astype(float)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    total = precision + recall
    return np.divide(2 * precision * recall, total, out=np.zeros_like(tp), where=total > 0)


def judge(lower, upper):
    """구간으로 판정한다. 하한 > 0이면 개선, 상한 < 0이면 저하, 그 외는 유의한 차이 없음."""
    if lower > 0:
        return IMPROVED
    if upper < 0:
        return DEGRADED
    return NO_DIFFERENCE


def pass_rule(status, verdict, score, threshold):
    """지표 하나의 통과 / 미달 / 불확정 (합격 규칙)."""
    if status is not None or threshold is None or score is None:
        return UNSETTLED
    if verdict == DEGRADED or score < threshold:
        return FAIL
    if verdict == IMPROVED and score >= threshold:
        return PASS
    return UNSETTLED


def overall_rule(results):
    """두 지표의 합격 결과로 종합 판정을 낸다. 하나라도 미달이면 미달, 모두 통과이면 통과, 그 외는 불확정."""
    outcomes = [r["pass"] for r in results]
    if FAIL in outcomes:
        return FAIL
    if outcomes and all(o == PASS for o in outcomes):
        return PASS
    return UNSETTLED


def compare(base_acc, ft_acc, base_pred, ft_pred, gold_oos, ci, thresholds):
    """base 실행 1벌과 파인튜닝 실행 S벌을 두 주 지표에서 비교한다.

    base_acc: 값 사례(n1개)의 `correct` 0/1. ft_acc: (S, n1).
    base_pred: 전체 사례(n2개)의 pred_oos 불리언. ft_pred: (S, n2). gold_oos: (n2,) 불리언.
    ci: alpha, B, rng_seed, min_units를 가진 dict. thresholds: axis1_min_abs, axis2_min_f1oos_abs.
    난수는 PCG64(rng_seed) 하나에서 나오고, 반복마다 순서대로 (1) 시드 S개, (2) 값 정확도용
    사례 n1개, (3) F1-OOS용 사례 n2개를 뽑는다. (3)은 표본에 정답 규격 외가 없으면 다시
    뽑으며, 처음 뽑은 것을 포함해 최대 50번까지 뽑는다.
    """
    base_acc = np.asarray(base_acc, dtype=float)
    ft_acc = np.asarray(ft_acc, dtype=float)
    base_pred = np.asarray(base_pred, dtype=bool)
    ft_pred = np.atleast_2d(np.asarray(ft_pred, dtype=bool))
    gold = np.asarray(gold_oos, dtype=bool)
    seeds = ft_acc.shape[0]
    n1, n2 = base_acc.shape[0], gold.shape[0]
    alpha, reps, min_units = ci["alpha"], int(ci["B"]), ci["min_units"]
    metrics = 2
    adjusted_alpha = alpha / metrics

    acc_status = None
    if n1 == 0:
        acc_status = UNDEFINED
    elif n1 < min_units:
        acc_status = SAMPLE_TOO_SMALL
    f1_status = None
    if not gold.any():
        f1_status = UNDEFINED
    elif n2 < min_units:
        f1_status = SAMPLE_TOO_SMALL

    ft_acc_score = float(ft_acc.mean(axis=1).mean()) if n1 else None
    base_acc_score = float(base_acc.mean()) if n1 else None
    if gold.any():
        ft_f1_score = float(f1_vector(ft_pred, gold).mean())
        base_f1_score = float(f1_vector(base_pred, gold)[0])
    else:
        ft_f1_score = base_f1_score = None

    rng = np.random.Generator(np.random.PCG64(int(ci["rng_seed"])))
    acc_diffs, f1_diffs = [], []
    f1_failed = False
    for _ in range(reps):
        slots = rng.integers(0, seeds, size=seeds)
        if acc_status is None:
            units = rng.integers(0, n1, size=n1)
            acc_diffs.append(ft_acc[slots][:, units].mean() - base_acc[units].mean())
        if f1_status is None:
            units = None
            for _try in range(MAX_REDRAWS):
                candidate = rng.integers(0, n2, size=n2)
                if gold[candidate].any():
                    units = candidate
                    break
            if units is None:
                f1_failed = True
            else:
                g = gold[units]
                ft_f1 = f1_vector(ft_pred[slots][:, units], g).mean()
                base_f1 = f1_vector(base_pred[units], g)[0]
                f1_diffs.append(ft_f1 - base_f1)
    if f1_failed:
        f1_status = RESAMPLING_FAILED

    def build(name, status, diffs, ft_score, base_score, threshold, units):
        """지표 하나의 결과 dict를 만든다. 점수, 보정 구간과 95% 구간, 판정, 합격 결과. 표본 부족 같은 상태가 있으면 구간 없이 불확정으로
        한다.
        """
        out = {"name": name, "ft_score": ft_score, "base_score": base_score, "units": units,
               "status": status, "ci_adjusted": None, "ci_95": None, "verdict": None,
               "threshold": threshold}
        if status is None:
            arr = np.asarray(diffs)
            low, high = np.percentile(arr, [100 * adjusted_alpha / 2, 100 * (1 - adjusted_alpha / 2)])
            low95, high95 = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            out["ci_adjusted"] = [float(low), float(high)]
            out["ci_95"] = [float(low95), float(high95)]
            out["verdict"] = judge(low, high)
        out["pass"] = pass_rule(status, out["verdict"], ft_score, threshold)
        return out

    axis1 = build("값 정확도", acc_status, acc_diffs, ft_acc_score, base_acc_score,
                  thresholds.get("axis1_min_abs"), n1)
    axis2 = build("F1-OOS", f1_status, f1_diffs, ft_f1_score, base_f1_score,
                  thresholds.get("axis2_min_f1oos_abs"), n2)
    return {"axis1": axis1, "axis2": axis2, "overall": overall_rule([axis1, axis2]),
            "bootstrap": {"B": reps, "alpha": alpha, "adjusted_alpha": adjusted_alpha,
                          "rng_seed": int(ci["rng_seed"]), "seed_runs": int(seeds)}}


__all__ = ["compare", "f1_vector", "f1_from_counts"]
