"""근거 유효율(A)과 계획 위반 표현(B): 검증 사례와 프레임워크 규칙 대조."""
import ast
import json
import random
import re
import unittest
from pathlib import Path

from scorer import checks, scoring_a, scoring_b

from .helpers import ROOT, dataset, gold_json, score_a

Q7 = "labeled100_081"
FRAMEWORK = ROOT / "externalData" / "framework" / "src" / "team_framework.py"


def context_of(case_id):
    return dataset().user_json(dataset().row_a(case_id))["context"]


class EvidenceCases(unittest.TestCase):
    """근거 검사 사례. 근거는 정오에 영향을 주지 않고 evidence_ok에만 반영된다."""

    def setUp(self):
        self.context = context_of(Q7)
        self.last = len(self.context) - 1

    def test_gold_is_valid(self):
        r = score_a(Q7, gold_json(Q7))
        self.assertTrue(r["evidence_ok"])
        self.assertEqual((r["evidence_items"], r["evidence_valid_items"]), (3, 3))

    def test_a19_end_beyond_the_turn(self):
        out = gold_json(Q7)
        out["facts"][0]["evidence"]["end"] = len(self.context[self.last]["content"]) + 5
        r = score_a(Q7, out)
        self.assertTrue(r["correct"])
        self.assertFalse(r["evidence_ok"])
        self.assertEqual((r["evidence_items"], r["evidence_valid_items"]), (3, 2))

    def test_a20_text_differs(self):
        out = gold_json(Q7)
        out["facts"][0]["evidence"]["text"] += "x"
        r = score_a(Q7, out)
        self.assertTrue(r["correct"])
        self.assertFalse(r["evidence_ok"])

    def test_a21_earlier_user_turn_with_matching_text(self):
        earlier = max(i for i, m in enumerate(self.context[:-1]) if m["role"] == "user")
        content = self.context[earlier]["content"]
        out = gold_json(Q7)
        out["facts"][0]["evidence"] = {"turn_index": earlier, "start": 0, "end": 3, "text": content[0:3]}
        r = score_a(Q7, out)
        self.assertTrue(r["correct"])
        self.assertFalse(r["evidence_ok"])           # 위치와 글자는 맞지만 현재 발화가 아니다

    def test_a22_assistant_turn(self):
        assistant = next(i for i, m in enumerate(self.context) if m["role"] == "assistant")
        content = self.context[assistant]["content"]
        out = gold_json(Q7)
        out["facts"][0]["evidence"] = {"turn_index": assistant, "start": 0, "end": 3, "text": content[0:3]}
        r = score_a(Q7, out)
        self.assertTrue(r["correct"])
        self.assertFalse(r["evidence_ok"])

    def test_a23_unmapped_fact_may_point_at_an_earlier_user_turn(self):
        earlier = max(i for i, m in enumerate(self.context[:-1]) if m["role"] == "user")
        content = self.context[earlier]["content"]
        out = gold_json(Q7)
        out["unmapped_facts"].append({"evidence": {"turn_index": earlier, "start": 0, "end": 3, "text": content[0:3]},
                                      "scope": "OUT_OF_SCHEMA"})
        r = score_a(Q7, out)
        self.assertFalse(r["correct"])               # 정답은 빈 배열이므로 오답
        self.assertTrue(r["evidence_ok"])            # unmapped_facts에는 마지막 턴 규칙을 적용하지 않는다

    def test_empty_start_end_and_negative_values(self):
        out = gold_json(Q7)
        ev = out["facts"][0]["evidence"]
        for start, end in ((ev["start"], ev["start"]), (-1, ev["end"]), (ev["end"], ev["start"])):
            bad = gold_json(Q7)
            bad["facts"][0]["evidence"].update(start=start, end=end)
            self.assertFalse(score_a(Q7, bad)["evidence_ok"], (start, end))

    def test_turn_index_out_of_range(self):
        for turn in (-1, len(self.context), len(self.context) + 10):
            out = gold_json(Q7)
            out["facts"][0]["evidence"]["turn_index"] = turn
            self.assertFalse(score_a(Q7, out)["evidence_ok"], turn)

    def test_output_without_evidence_items_is_valid(self):
        empty = {"facts": [], "intents": [], "relations": [], "unmapped_facts": []}
        r = score_a(Q7, empty)
        self.assertTrue(r["evidence_ok"])
        self.assertEqual(r["evidence_items"], 0)

    def test_none_when_schema_fails_or_missing(self):
        self.assertIsNone(score_a(Q7, "not json")["evidence_ok"])
        self.assertIsNone(score_a(Q7, None)["evidence_ok"])
        broken = gold_json(Q7)
        del broken["facts"][0]["evidence"]
        self.assertIsNone(score_a(Q7, broken)["evidence_ok"])

    def test_evidence_of_every_gold_is_valid(self):
        data = dataset()
        total = 0
        for case_id in data.ids:
            r = score_a(case_id, gold_json(case_id))
            self.assertTrue(r["evidence_ok"], case_id)
            total += r["evidence_items"]
        self.assertEqual(total, 158)


class EvidenceSummary(unittest.TestCase):
    def test_summary_counts(self):
        data = dataset()
        cases = [Q7, "labeled100_082"]
        rows, results, golds = [], [], []
        for index, case_id in enumerate(cases):
            out = gold_json(case_id)
            if index == 0:
                out["facts"][0]["evidence"]["text"] += "x"     # 첫 사례는 근거 하나가 틀리다
            gold = data.gold_a(case_id)
            result = score_a(case_id, out)
            rows.append(scoring_a.score_row(case_id, "x", gold, result))
            results.append(result)
            golds.append(gold)
        evidence = scoring_a.summarize(rows, results, golds, 30)["evidence"]
        self.assertEqual(evidence["outputs_checked"], 2)
        self.assertEqual(evidence["outputs_valid"], 1)
        self.assertEqual(evidence["output_valid_rate"], 0.5)
        self.assertEqual(evidence["items"], evidence["valid_items"] + 1)
        self.assertAlmostEqual(evidence["item_valid_rate"], evidence["valid_items"] / evidence["items"])

    def test_rates_are_none_when_nothing_was_checked(self):
        data = dataset()
        gold = data.gold_a(Q7)
        result = score_a(Q7, None)
        row = scoring_a.score_row(Q7, "x", gold, result)
        evidence = scoring_a.summarize([row], [result], [gold], 30)["evidence"]
        self.assertEqual((evidence["outputs_checked"], evidence["output_valid_rate"], evidence["item_valid_rate"]),
                         (0, None, None))


class CompletionClaim(unittest.TestCase):
    """계획 위반 표현 검사 사례."""

    def test_b7_claims_completion(self):
        for text in ("입력하신 내용이 저장되었습니다.", "최종 확정했습니다.", "작성이 완료됐어요.", "완료되셨습니다"):
            self.assertTrue(checks.has_completion_claim(text, "NEXT"), text)

    def test_b8_not_a_claim(self):
        for text in ("아직 최종 완료는 아닙니다.", "저장 전에 확인해 주세요.", "확정하실 건가요?"):
            self.assertFalse(checks.has_completion_claim(text, "REVIEW_SUMMARY"), text)

    def test_b9_empty_and_missing(self):
        for text in ("", "   ", None, 5):
            self.assertFalse(checks.has_completion_claim(text, "NEXT"))

    def test_complete_action_is_exempt(self):
        self.assertFalse(checks.has_completion_claim("저장되었습니다.", "COMPLETE"))

    def test_row_flag_and_summary(self):
        data = dataset()
        ids = data.ids[:3]
        outputs = {ids[0]: "입력하신 내용이 저장되었습니다.", ids[1]: "다음 질문입니다.", ids[2]: 7}
        small = _SubsetData(data, ids)
        scorer = scoring_b.SentenceScorer(scoring_b.build_corpus(data))
        rows = scoring_b.score_run(small, outputs, scorer)
        self.assertEqual([r["completion_claim"] for r in rows], [True, False, False])
        self.assertTrue(rows[2]["missing"])
        summary = scoring_b.summarize(rows, small, scorer, 18692)
        self.assertEqual(summary["completion_claims"], 1)
        self.assertAlmostEqual(summary["completion_claim_rate"], 0.5)     # 결측 1건은 분모에서 뺀다

    def test_gold_sentences_and_chatbot_sentences_have_no_false_positive(self):
        data = dataset()
        for case_id in data.ids:
            self.assertFalse(checks.has_completion_claim(data.reference(case_id), data.plan_action(case_id)), case_id)
        for sentence in scoring_b.build_corpus(data):
            self.assertFalse(checks.has_completion_claim(sentence, "NEXT"), sentence)


class _SubsetData:
    """사례 일부만 가진 데이터. score_run과 summarize에 필요한 부분만 흉내 낸다."""

    def __init__(self, data, ids):
        self._data, self.ids = data, ids

    def reference(self, case_id):
        return self._data.reference(case_id)

    def references(self):
        return [self._data.reference(i) for i in self.ids]

    def plan_action(self, case_id):
        return self._data.plan_action(case_id)


@unittest.skipUnless(FRAMEWORK.exists(), "externalData/framework 가 없으면 건너뛴다")
class MatchesTheFramework(unittest.TestCase):
    """프레임워크의 실제 함수와 정규식을 소스에서 떼어 와 우리 구현과 대조한다(프레임워크는 실행하지 않는다)."""

    @classmethod
    def setUpClass(cls):
        source = FRAMEWORK.read_text(encoding="utf-8")
        namespace = {}
        exec("class ContractError(ValueError): pass", namespace)
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name == "evidence_ok":
                exec(ast.get_source_segment(source, node), namespace)
        cls.theirs = staticmethod(namespace["evidence_ok"])   # 인스턴스로 부를 때 self가 끼지 않게 한다
        cls.pattern = re.search(r"re\.search\(r'(\(완료\|저장\|확정\)[^']*)'", source).group(1)

    def test_same_regular_expression(self):
        self.assertEqual(self.pattern, checks.COMPLETION_CLAIM.pattern)

    def test_same_evidence_verdict_on_many_corruptions(self):
        data = dataset()
        rng = random.Random(7)
        compared = disagreements = 0
        for case_id in data.ids:
            context = context_of(case_id)
            gold = gold_json(case_id)
            for fact in gold["facts"] + gold["intents"]:
                for _ in range(40):
                    ev = dict(fact["evidence"])
                    kind = rng.choice(("none", "turn", "start", "end", "text", "range", "retarget", "retarget"))
                    if kind == "retarget":
                        # 아무 턴(assistant 포함)의 실제 글자를 정확히 가리키는 근거. 위치와 글자는 맞고
                        # 턴의 role과 현재 턴 여부만 다르므로 ②와 ⑤를 구별해 낸다.
                        turn = rng.randrange(len(context))
                        content = context[turn]["content"]
                        start = rng.randrange(len(content))
                        end = rng.randint(start + 1, min(len(content), start + 6))
                        ev = {"turn_index": turn, "start": start, "end": end, "text": content[start:end]}
                    elif kind == "turn":
                        ev["turn_index"] = rng.randint(-2, len(context) + 1)
                    elif kind == "start":
                        ev["start"] += rng.randint(-3, 3)
                    elif kind == "end":
                        ev["end"] += rng.randint(-3, 3)
                    elif kind == "text":
                        ev["text"] = ev["text"][:-1] + rng.choice(("가", "x", ""))
                    elif kind == "range":
                        ev["start"], ev["end"] = rng.randint(0, 5), rng.randint(0, 8)
                    try:
                        self.theirs(ev, context)
                        expected = True
                    except Exception:
                        expected = False
                    got = checks.evidence_valid(ev, context, must_be_last_turn=False)
                    compared += 1
                    disagreements += int(expected != got)
                    # 마지막 턴 규칙(프레임워크의 validate에서 따로 검사하는 부분)
                    last_expected = expected and ev["turn_index"] == len(context) - 1
                    self.assertEqual(checks.evidence_valid(ev, context, must_be_last_turn=True), last_expected)
        self.assertGreater(compared, 5000)
        self.assertEqual(disagreements, 0)


if __name__ == "__main__":
    unittest.main()
