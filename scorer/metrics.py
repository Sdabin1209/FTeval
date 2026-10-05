"""채점과 비교에서 함께 쓰는 작은 지표 함수들."""
import numpy as np


def f1_from_counts(tp, fp, fn):
    """규격 외 클래스의 F1. 정답 규격 외가 없으면 None("정의 불가")."""
    if tp + fn == 0:
        return None
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def prf(tp, pred_total, gold_total):
    """정밀도, 재현율, F1. 분모가 0이면 그 항은 0이다."""
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def distribution(values):
    """최소, 25·50·75 백분위(선형 보간), 최대. 빈 목록이면 None."""
    if not len(values):
        return None
    arr = np.asarray(values, dtype=float)
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    return {"min": float(arr.min()), "p25": float(q25), "p50": float(q50),
            "p75": float(q75), "max": float(arr.max())}


def population_sd(values):
    """sqrt((1/S) * sum((x - 평균)^2)). 빈 목록이면 None."""
    if not len(values):
        return None
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean((arr - arr.mean()) ** 2)))
