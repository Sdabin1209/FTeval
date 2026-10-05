"""지표와 base 대 ft 비교."""
import unittest
from unittest import mock

import numpy as np

from scorer import stats
from scorer.metrics import distribution, f1_from_counts, population_sd, prf

CI = {"alpha": 0.05, "B": 400, "rng_seed": 0, "min_units": 30}
THR = {"axis1_min_abs": 0.90, "axis2_min_f1oos_abs": 0.80}


class MetricValues(unittest.TestCase):
    def test_f1_oos_of_the_spec_example(self):
        # 손으로 계산한 예: base TP=2 FP=2 FN=2 -> 0.5, ft TP=3 FP=0 FN=1 -> P=1, R=0.75 -> 0.857
        self.assertAlmostEqual(f1_from_counts(2, 2, 2), 0.5)
        self.assertAlmostEqual(f1_from_counts(3, 0, 1), 2 * 1 * 0.75 / 1.75)

    def test_conventions(self):
        self.assertIsNone(f1_from_counts(0, 3, 0))        # 정답 규격 외가 없음: 정의 불가
        self.assertEqual(f1_from_counts(0, 0, 2), 0.0)    # TP+FP = 0 -> P = 0
        self.assertEqual(f1_from_counts(0, 2, 2), 0.0)    # P + R = 0
        self.assertEqual(prf(0, 0, 0), (0.0, 0.0, 0.0))

    def test_f1_vector_matches_counts(self):
        rng = np.random.default_rng(1)
        gold = rng.random(60) < 0.2
        pred = rng.random((4, 60)) < 0.3
        vec = stats.f1_vector(pred, gold)
        for i in range(4):
            tp = int((pred[i] & gold).sum())
            fp = int((pred[i] & ~gold).sum())
            fn = int((~pred[i] & gold).sum())
            self.assertAlmostEqual(vec[i], f1_from_counts(tp, fp, fn))

    def test_distribution_and_sd(self):
        d = distribution([1, 2, 3, 4, 5])
        self.assertEqual((d["min"], d["p25"], d["p50"], d["p75"], d["max"]), (1, 2, 3, 4, 5))
        self.assertAlmostEqual(distribution([0, 1])["p25"], 0.25)       # 선형 보간
        self.assertAlmostEqual(population_sd([1, 3]), 1.0)               # 분모는 S이고 S-1이 아니다
        self.assertIsNone(distribution([]))


class PassRules(unittest.TestCase):
    def test_rule_order(self):
        self.assertEqual(stats.pass_rule(None, stats.IMPROVED, 0.95, 0.9), stats.PASS)
        self.assertEqual(stats.pass_rule(None, stats.IMPROVED, 0.85, 0.9), stats.FAIL)
        self.assertEqual(stats.pass_rule(None, stats.DEGRADED, 0.99, 0.9), stats.FAIL)
        self.assertEqual(stats.pass_rule(None, stats.NO_DIFFERENCE, 0.99, 0.9), stats.UNSETTLED)
        self.assertEqual(stats.pass_rule(None, stats.NO_DIFFERENCE, 0.5, 0.9), stats.FAIL)
        for status in (stats.SAMPLE_TOO_SMALL, stats.RESAMPLING_FAILED, stats.UNDEFINED):
            self.assertEqual(stats.pass_rule(status, None, 0.1, 0.9), stats.UNSETTLED)

    def test_overall(self):
        p, f, u = {"pass": stats.PASS}, {"pass": stats.FAIL}, {"pass": stats.UNSETTLED}
        self.assertEqual(stats.overall_rule([p, p]), stats.PASS)
        self.assertEqual(stats.overall_rule([p, f]), stats.FAIL)
        self.assertEqual(stats.overall_rule([u, f]), stats.FAIL)
        self.assertEqual(stats.overall_rule([p, u]), stats.UNSETTLED)


def make(n1=40, n2=60, oos=6, seeds=5, base_good=0.0, ft_good=1.0, seed=3):
    """합성 점수 배열. `good`은 실행이 맞히는 사례의 비율이다."""
    rng = np.random.default_rng(seed)
    gold = np.zeros(n2, dtype=bool)
    gold[:oos] = True
    base_acc = (rng.random(n1) < base_good).astype(float)
    ft_acc = (rng.random((seeds, n1)) < ft_good).astype(float)
    base_pred = np.where(rng.random(n2) < base_good, gold, ~gold)
    ft_pred = np.where(rng.random((seeds, n2)) < ft_good, gold, ~gold)
    return base_acc, ft_acc, base_pred, ft_pred, gold


class Compare(unittest.TestCase):
    def run_compare(self, *arrays, ci=CI, thresholds=THR):
        return stats.compare(*arrays, ci=ci, thresholds=thresholds)

    def test_clear_improvement_passes(self):
        res = self.run_compare(*make(base_good=0.0, ft_good=1.0))
        a1, a2 = res["axis1"], res["axis2"]
        self.assertEqual((a1["verdict"], a1["pass"]), (stats.IMPROVED, stats.PASS))
        self.assertEqual((a2["verdict"], a2["pass"]), (stats.IMPROVED, stats.PASS))
        self.assertEqual(a1["ci_adjusted"], [1.0, 1.0])
        self.assertEqual(res["overall"], stats.PASS)
        self.assertAlmostEqual(res["bootstrap"]["adjusted_alpha"], 0.025)

    def test_same_runs_are_not_different(self):
        arrays = make(base_good=1.0, ft_good=1.0)
        res = self.run_compare(*arrays)
        self.assertEqual(res["axis1"]["verdict"], stats.NO_DIFFERENCE)
        self.assertEqual(res["axis1"]["pass"], stats.UNSETTLED)      # 점수는 1.0이지만 유의한 차이가 없음
        self.assertEqual(res["overall"], stats.UNSETTLED)

    def test_degradation_fails(self):
        res = self.run_compare(*make(base_good=1.0, ft_good=0.0))
        self.assertEqual(res["axis1"]["verdict"], stats.DEGRADED)
        self.assertEqual(res["overall"], stats.FAIL)

    def test_improved_but_under_threshold_fails(self):
        arrays = make(base_good=0.0, ft_good=0.5)
        res = self.run_compare(*arrays)
        self.assertEqual(res["axis1"]["verdict"], stats.IMPROVED)
        self.assertLess(res["axis1"]["ft_score"], 0.9)
        self.assertEqual(res["axis1"]["pass"], stats.FAIL)

    def test_small_sample_gives_no_interval(self):
        res = self.run_compare(*make(n1=20, n2=25))
        for key in ("axis1", "axis2"):
            self.assertEqual(res[key]["status"], stats.SAMPLE_TOO_SMALL)
            self.assertIsNone(res[key]["ci_adjusted"])
            self.assertEqual(res[key]["pass"], stats.UNSETTLED)
        self.assertEqual(res["overall"], stats.UNSETTLED)

    def test_min_units_comes_from_the_config(self):
        res = self.run_compare(*make(n1=20, n2=25), ci=dict(CI, min_units=10))
        self.assertIsNone(res["axis1"]["status"])

    def test_no_gold_out_of_scope_case(self):
        res = self.run_compare(*make(oos=0))
        self.assertEqual(res["axis2"]["status"], stats.UNDEFINED)
        self.assertIsNone(res["axis2"]["ft_score"])
        self.assertIsNotNone(res["axis1"]["ci_adjusted"])

    def test_resampling_failure(self):
        with mock.patch.object(stats, "MAX_REDRAWS", 0):
            res = self.run_compare(*make())
        self.assertEqual(res["axis2"]["status"], stats.RESAMPLING_FAILED)
        self.assertEqual(res["axis2"]["pass"], stats.UNSETTLED)
        self.assertIsNotNone(res["axis1"]["ci_adjusted"])

    def test_deterministic_and_seed_dependent(self):
        arrays = make(base_good=0.6, ft_good=0.75)
        first = self.run_compare(*arrays)
        again = self.run_compare(*arrays)
        self.assertEqual(first, again)
        other = self.run_compare(*arrays, ci=dict(CI, rng_seed=1))
        self.assertNotEqual(first["axis1"]["ci_95"], other["axis1"]["ci_95"])

    def test_single_seed_works(self):
        res = self.run_compare(*make(seeds=1, base_good=0.0, ft_good=1.0))
        self.assertEqual(res["bootstrap"]["seed_runs"], 1)
        self.assertEqual(res["axis1"]["verdict"], stats.IMPROVED)

    def test_point_scores_are_seed_means(self):
        arrays = make(base_good=0.3, ft_good=0.8, seeds=3)
        res = self.run_compare(*arrays)
        base_acc, ft_acc, base_pred, ft_pred, gold = arrays
        self.assertAlmostEqual(res["axis1"]["ft_score"], ft_acc.mean(axis=1).mean())
        self.assertAlmostEqual(res["axis1"]["base_score"], base_acc.mean())
        per_seed = [f1_from_counts(int((p & gold).sum()), int((p & ~gold).sum()), int((~p & gold).sum()))
                    for p in ft_pred]
        self.assertAlmostEqual(res["axis2"]["ft_score"], sum(per_seed) / 3)

    def test_matches_an_independent_loop(self):
        """문서에 적힌 난수 순서대로 단순 반복문으로 부트스트랩을 다시 계산한다."""
        base_acc, ft_acc, base_pred, ft_pred, gold = make(base_good=0.5, ft_good=0.7, seeds=4)
        ci = dict(CI, B=60)
        res = self.run_compare(base_acc, ft_acc, base_pred, ft_pred, gold, ci=ci)
        rng = np.random.Generator(np.random.PCG64(0))
        n1, n2, seeds = len(base_acc), len(gold), ft_acc.shape[0]
        acc_diffs, f1_diffs = [], []
        for _ in range(60):
            slots = rng.integers(0, seeds, size=seeds)
            units = rng.integers(0, n1, size=n1)
            ft = np.mean([ft_acc[s][units].mean() for s in slots])
            acc_diffs.append(ft - base_acc[units].mean())
            while True:
                units2 = rng.integers(0, n2, size=n2)
                if gold[units2].any():
                    break

            def f1(pred):
                g = gold[units2]
                p = pred[units2]
                return f1_from_counts(int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum()))
            f1_diffs.append(np.mean([f1(ft_pred[s]) for s in slots]) - f1(base_pred))
        for name, diffs in (("axis1", acc_diffs), ("axis2", f1_diffs)):
            low, high = np.percentile(diffs, [1.25, 98.75])
            self.assertAlmostEqual(res[name]["ci_adjusted"][0], low)
            self.assertAlmostEqual(res[name]["ci_adjusted"][1], high)
            low, high = np.percentile(diffs, [2.5, 97.5])
            self.assertAlmostEqual(res[name]["ci_95"][0], low)
            self.assertAlmostEqual(res[name]["ci_95"][1], high)


if __name__ == "__main__":
    unittest.main()
