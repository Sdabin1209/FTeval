"""역할 A 검증 사례: 손으로 계산한 기대값을 정해 두고 채점기가 그대로 내는지 본다."""
import json
import unittest

from scorer import scoring_a

from .helpers import dataset, gold_json, score_a

Q7 = "labeled100_081"      # Q7: frequency=4 ANSWERED, unit MISSING, does_not_drink=false ANSWERED
Q6_1 = "labeled100_053"    # Q6_1: usage_days=0


def fact(output, field):
    return next(f for f in output["facts"] if f["field"] == field)


class CasesQ7(unittest.TestCase):
    def test_gold_is_what_the_spec_describes(self):
        gold = gold_json(Q7)
        self.assertEqual([(f["field"], f["semantic_status"], f["precision"], f["value"]) for f in gold["facts"]],
                         [("frequency", "ANSWERED", "exact", 4), ("unit", "MISSING", "unspecified", None),
                          ("does_not_drink", "ANSWERED", "exact", False)])
        self.assertEqual((gold["intents"], gold["relations"], gold["unmapped_facts"]), ([], [], []))

    def test_01_identical(self):
        r = score_a(Q7, gold_json(Q7))
        self.assertTrue(r["correct"] and r["core_correct"] and r["values_ok"])

    def test_02_guessed_unit(self):
        out = gold_json(Q7)
        u = fact(out, "unit")
        u.update(value="week", semantic_status="ANSWERED", precision="exact")
        r = score_a(Q7, out)
        self.assertFalse(r["correct"])
        self.assertFalse(r["core_correct"])

    def test_03_precision_only(self):
        out = gold_json(Q7)
        fact(out, "unit")["precision"] = "exact"
        r = score_a(Q7, out)
        self.assertFalse(r["correct"])
        self.assertTrue(r["core_correct"])

    def test_04_float_four(self):
        out = gold_json(Q7)
        raw = json.dumps(out, ensure_ascii=False).replace('"value": 4}', '"value": 4.0}')
        self.assertIn('"value": 4.0}', raw)
        self.assertTrue(score_a(Q7, raw)["correct"])

    def test_05_missing_fact(self):
        out = gold_json(Q7)
        out["facts"] = [f for f in out["facts"] if f["field"] != "does_not_drink"]
        self.assertFalse(score_a(Q7, out)["correct"])

    def test_06_evidence_start_differs(self):
        out = gold_json(Q7)
        out["facts"][0]["evidence"]["start"] += 1
        self.assertTrue(score_a(Q7, out)["correct"])

    def test_07_extra_refused_intent(self):
        out = gold_json(Q7)
        out["intents"].append({"item_id": "Q7", "intent": "REFUSED", "evidence": out["facts"][0]["evidence"]})
        r = score_a(Q7, out)
        self.assertFalse(r["correct"])
        self.assertTrue(r["pred_oos"])
        data = dataset()
        gold = data.gold_a(Q7)
        row = scoring_a.score_row(Q7, "x", gold, r)
        summary = scoring_a.summarize([row], [r], [gold], 30)
        self.assertEqual(summary["diagnostics"]["value_cases_judged_out_of_scope"], 1)

    def test_08_code_block(self):
        raw = "```json\n" + json.dumps(gold_json(Q7), ensure_ascii=False) + "\n```"
        r = score_a(Q7, raw)
        self.assertFalse(r["parse_ok"])
        self.assertFalse(r["schema_ok"])
        self.assertFalse(r["correct"])
        self.assertIsNone(r["values_ok"])
        self.assertTrue(r["pred_oos"])     # 정답은 규격 외가 아니므로 실패가 "규격 외"로 계산된다

    def test_09_duplicate_field(self):
        out = gold_json(Q7)
        out["facts"].append(dict(out["facts"][0]))
        r = score_a(Q7, out)
        self.assertTrue(r["schema_ok"])
        self.assertFalse(r["values_ok"])
        self.assertFalse(r["correct"])

    def test_10_evidence_key_missing(self):
        out = gold_json(Q7)
        del out["facts"][0]["evidence"]
        r = score_a(Q7, out)
        self.assertTrue(r["parse_ok"])
        self.assertFalse(r["schema_ok"])
        self.assertFalse(r["correct"])

    def test_11_precision_range(self):
        out = gold_json(Q7)
        fact(out, "frequency")["precision"] = "range"
        r = score_a(Q7, out)
        self.assertTrue(r["values_ok"])
        self.assertFalse(r["correct"])
        self.assertTrue(r["core_correct"])

    def test_12_answer_intent(self):
        out = gold_json(Q7)
        out["intents"].append({"item_id": "Q7", "intent": "ANSWER", "evidence": out["facts"][0]["evidence"]})
        r = score_a(Q7, out)
        self.assertTrue(r["values_ok"])
        self.assertFalse(r["correct"])
        self.assertFalse(r["pred_oos"])

    def test_13_intent_outside_allowed_values(self):
        out = gold_json(Q7)
        out["intents"].append({"item_id": "Q7", "intent": "MAYBE", "evidence": out["facts"][0]["evidence"]})
        r = score_a(Q7, out)
        self.assertTrue(r["schema_ok"])
        self.assertFalse(r["values_ok"])
        self.assertFalse(r["correct"])

    def test_14_new_information_relation(self):
        out = gold_json(Q7)
        out["relations"].append({"item_id": "Q7", "relation": "NEW_INFORMATION", "correction_target": None,
                                 "evidence": out["facts"][0]["evidence"]})
        r = score_a(Q7, out)
        self.assertTrue(r["values_ok"])
        self.assertFalse(r["correct"])

    def test_15_unmapped_fact(self):
        out = gold_json(Q7)
        out["unmapped_facts"].append({"evidence": out["facts"][0]["evidence"], "scope": "OUT_OF_SCHEMA"})
        r = score_a(Q7, out)
        self.assertTrue(r["schema_ok"] and r["values_ok"])
        self.assertFalse(r["correct"])

    def test_16_unmapped_scope_key_missing(self):
        out = gold_json(Q7)
        out["unmapped_facts"].append({"evidence": out["facts"][0]["evidence"]})
        r = score_a(Q7, out)
        self.assertFalse(r["schema_ok"])
        self.assertFalse(r["correct"])

    def test_17_unmapped_scope_other(self):
        out = gold_json(Q7)
        out["unmapped_facts"].append({"evidence": out["facts"][0]["evidence"], "scope": "OTHER"})
        r = score_a(Q7, out)
        self.assertTrue(r["schema_ok"])
        self.assertFalse(r["values_ok"])
        self.assertFalse(r["correct"])


class CaseQ6_1(unittest.TestCase):
    def test_18_category_string_for_integer_field(self):
        out = gold_json(Q6_1)
        self.assertEqual(out["facts"][0]["field"], "usage_days")
        out["facts"][0]["value"] = "days_1_2"
        r = score_a(Q6_1, out)
        self.assertTrue(r["schema_ok"])
        self.assertFalse(r["values_ok"])
        self.assertFalse(r["correct"])

    def test_gold_itself(self):
        self.assertTrue(score_a(Q6_1, gold_json(Q6_1))["correct"])


class OutOfScopeGold(unittest.TestCase):
    def test_refused_gold_is_a_tp_and_not_a_value_case(self):
        data = dataset()
        refused = [i for i in data.ids if data.gold_a(i)["oos"]]
        self.assertEqual(len(refused), 6)
        case = refused[0]
        r = score_a(case, gold_json(case))
        self.assertTrue(r["pred_oos"])
        self.assertTrue(r["correct"])   # 이 사례들의 `correct`는 의미가 없고 값 정확도에서 제외된다


if __name__ == "__main__":
    unittest.main()
