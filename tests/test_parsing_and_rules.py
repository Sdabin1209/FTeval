"""엄격한 JSON 읽기, 스키마와 값 규칙, 입력 파일 읽기."""
import json
import tempfile
import unittest
from pathlib import Path

from scorer import schema
from scorer.dataio import InputError, read_outputs
from scorer.strictjson import StrictJSONError, parse_strict_object

from .helpers import dataset, gold_json, score_a

EMPTY = '{"facts":[],"intents":[],"relations":[],"unmapped_facts":[]}'


class StrictJson(unittest.TestCase):
    def test_accepts_plain_object_with_outer_whitespace(self):
        self.assertEqual(parse_strict_object(" \n" + EMPTY + "\t\r\n")["facts"], [])

    def test_rejections(self):
        bad = {
            "duplicate key": '{"a":1,"a":2}',
            "nested duplicate key": '{"a":{"b":1,"b":2}}',
            "NaN": '{"a":NaN}',
            "Infinity": '{"a":Infinity}',
            "-Infinity": '{"a":-Infinity}',
            "overflow": '{"a":1e999}',
            "BOM": "﻿" + EMPTY,
            "trailing text": EMPTY + " done",
            "code block": "```json\n" + EMPTY + "\n```",
            "top-level list": "[]",
            "top-level number": "1",
            "empty": "",
        }
        for name, text in bad.items():
            with self.subTest(name):
                with self.assertRaises(StrictJSONError):
                    parse_strict_object(text)

    def test_non_string(self):
        with self.assertRaises(StrictJSONError):
            parse_strict_object(None)


def base_output():
    return json.loads(EMPTY)


def fact(item_id, field, value, status="ANSWERED", precision="exact"):
    return {"item_id": item_id, "field": field, "value": value, "semantic_status": status,
            "precision": precision, "evidence": {"turn_index": 1, "start": 0, "end": 1, "text": "x"}}


QIDS = frozenset(schema.ITEM_IDS)


def value_problems(*facts):
    out = base_output()
    out["facts"] = list(facts)
    assert not schema.schema_errors(out), schema.schema_errors(out)
    return schema.value_errors(out, QIDS)


class ValueRules(unittest.TestCase):
    def ok(self, *facts):
        self.assertEqual(value_problems(*facts), [])

    def bad(self, *facts):
        self.assertNotEqual(value_problems(*facts), [])

    def test_types(self):
        self.ok(fact("Q1.D01", "diagnosed", True))
        self.bad(fact("Q1.D01", "diagnosed", 1))
        self.bad(fact("Q1.D01", "diagnosed", "true"))
        self.ok(fact("Q8_1", "answer", 7))
        self.bad(fact("Q8_1", "answer", 8))
        self.bad(fact("Q8_1", "answer", -1))
        self.bad(fact("Q8_1", "answer", 1.0))
        self.bad(fact("Q8_1", "answer", True))
        self.ok(fact("Q7", "frequency", 1.5))
        self.bad(fact("Q7", "frequency", 0))
        self.bad(fact("Q7", "frequency", True))
        self.ok(fact("Q7", "unit", "week"))
        self.bad(fact("Q7", "unit", "day"))
        self.ok(fact("Q8_2", "minutes", 59))
        self.bad(fact("Q8_2", "minutes", 60))
        self.ok(fact("Q4_1", "daily_count", 0.5))
        self.bad(fact("Q4_1", "daily_count", -0.5))
        self.ok(fact("Q3", "answer", "unknown"))
        self.bad(fact("Q3", "answer", "maybe"))

    def test_fields_and_items(self):
        self.bad(fact("Q2.D01", "disease", "x"))
        self.bad(fact("Q99", "answer", True))
        self.bad(fact("Q1.D12", "diagnosed", True))

    def test_amounts_index_notation(self):
        self.ok(fact("Q7_1", "amounts[0].beverage", "소주"), fact("Q7_1", "amounts[12].amount", 2.5))
        self.bad(fact("Q7_1", "amounts[01].beverage", "소주"))
        self.bad(fact("Q7_1", "amounts[-1].beverage", "소주"))
        self.bad(fact("Q7_1", "amounts[0].volume", 1))
        self.bad(fact("Q7_1", "amounts[0].unit", "L"))
        self.ok(fact("Q7_2", "amounts[0].unit", "cc"))

    def test_q6_1_is_any_integer(self):
        self.ok(fact("Q6_1", "usage_days", 45))
        self.bad(fact("Q6_1", "usage_days", "daily"))

    def test_null_rules(self):
        self.ok(fact("Q7", "unit", None, "MISSING", "unspecified"))
        self.bad(fact("Q7", "unit", "week", "MISSING", "unspecified"))
        self.bad(fact("Q7", "unit", None, "ANSWERED"))

    def test_duplicates(self):
        self.bad(fact("Q1.D01", "diagnosed", True), fact("Q1.D01", "diagnosed", False))

    def test_enums_of_status_and_precision(self):
        self.bad(fact("Q1.D01", "diagnosed", True, status="DONE"))
        self.bad(fact("Q1.D01", "diagnosed", True, precision="vague"))
        for p in schema.PRECISION:
            self.ok(fact("Q1.D01", "diagnosed", True, precision=p))

    def test_item_must_be_in_the_input_questionnaire(self):
        out = base_output()
        out["facts"] = [fact("Q1.D01", "diagnosed", True)]
        self.assertNotEqual(schema.value_errors(out, frozenset(["Q3"])), [])


class SchemaChecks(unittest.TestCase):
    def test_empty_output_is_fine(self):
        self.assertEqual(schema.schema_errors(base_output()), [])

    def test_top_level_keys(self):
        for mutate in (lambda o: o.pop("facts"), lambda o: o.update(extra=[]), lambda o: o.update(facts={})):
            out = base_output()
            mutate(out)
            self.assertNotEqual(schema.schema_errors(out), [])

    def test_extra_key_in_item(self):
        out = base_output()
        f = fact("Q3", "answer", "yes")
        f["note"] = "x"
        out["facts"] = [f]
        self.assertNotEqual(schema.schema_errors(out), [])

    def test_evidence_types(self):
        out = base_output()
        f = fact("Q3", "answer", "yes")
        f["evidence"]["start"] = True
        out["facts"] = [f]
        self.assertNotEqual(schema.schema_errors(out), [])
        f["evidence"]["start"] = 0
        f["evidence"]["text"] = 5
        self.assertNotEqual(schema.schema_errors(out), [])

    def test_correction_target(self):
        evidence = {"turn_index": 1, "start": 0, "end": 1, "text": "x"}
        target = {"item_id": "Q8_1", "field": "answer", "turn_index": 3}
        for target_value, expected_ok in ((target, True), (None, True), ({"item_id": "Q8_1"}, False),
                                          (dict(target, extra=1), False), (dict(target, turn_index=1.5), False)):
            out = base_output()
            out["relations"] = [{"item_id": "Q8_1", "relation": "CORRECTION",
                                 "correction_target": target_value, "evidence": evidence}]
            self.assertEqual(schema.schema_errors(out) == [], expected_ok, target_value)
        out = base_output()
        out["relations"] = [{"item_id": "Q8_1", "relation": "CORRECTION", "evidence": evidence}]
        self.assertNotEqual(schema.schema_errors(out), [])   # 키가 반드시 있어야 한다


class WholeGold(unittest.TestCase):
    def test_every_gold_scores_as_correct(self):
        data = dataset()
        for case_id in data.ids:
            r = score_a(case_id, gold_json(case_id))
            self.assertTrue(r["correct"] and r["values_ok"], case_id)
            self.assertEqual(r["pred_oos"], data.gold_a(case_id)["oos"], case_id)

    def test_gold_evidence_points_at_the_text(self):
        data = dataset()
        for case_id in data.ids:
            context = data.user_json(data.row_a(case_id))["context"]
            for f in gold_json(case_id)["facts"]:
                e = f["evidence"]
                self.assertEqual(context[e["turn_index"]]["content"][e["start"]:e["end"]], e["text"])


class OutputFiles(unittest.TestCase):
    def write(self, tmp, text):
        path = Path(tmp) / "out.jsonl"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_known_ids_ignores_unknown_and_blank_lines(self):
        ids = {"a", "b"}
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, '{"id":"a","raw_output":"x","extra":1}\n\n{"id":"zzz","raw_output":"y"}\n'
                                   '{"id":5,"raw_output":"y"}\n')
            values, warnings = read_outputs(path, "raw_output", ids)
        self.assertEqual(values, {"a": "x"})
        self.assertEqual(len(warnings), 2)

    def test_errors(self):
        ids = {"a"}
        cases = {
            "not json": 'oops\n',
            "not an object": '[1]\n',
            "no id": '{"raw_output":"x"}\n',
            "no key": '{"id":"a"}\n',
            "duplicate id": '{"id":"a","raw_output":"x"}\n{"id":"a","raw_output":"y"}\n',
            "duplicate unknown id": '{"id":"q","raw_output":"x"}\n{"id":"q","raw_output":"y"}\n',
        }
        for name, text in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(InputError):
                    read_outputs(self.write(tmp, text), "raw_output", ids)

    def test_missing_ids_are_simply_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            values, _ = read_outputs(self.write(tmp, ""), "raw_output", {"a"})
        self.assertEqual(values, {})

    def test_non_string_value_is_kept_for_the_scorer_to_treat_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            values, _ = read_outputs(self.write(tmp, '{"id":"a","raw_output":7}\n'), "raw_output", {"a"})
        self.assertEqual(values, {"a": 7})


class MissingOutput(unittest.TestCase):
    def test_missing_is_scored_as_wrong_and_out_of_scope_flipped(self):
        data = dataset()
        value_case = next(i for i in data.ids if not data.gold_a(i)["oos"])
        oos_case = next(i for i in data.ids if data.gold_a(i)["oos"])
        for raw in (None, 7):
            r = score_a(value_case, raw)
            self.assertTrue(r["missing"] and not r["parse_ok"] and not r["schema_ok"])
            self.assertIsNone(r["values_ok"])
            self.assertFalse(r["correct"])
            self.assertTrue(r["pred_oos"])
            self.assertFalse(score_a(oos_case, raw)["pred_oos"])


if __name__ == "__main__":
    unittest.main()
