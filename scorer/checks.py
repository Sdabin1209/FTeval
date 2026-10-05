"""문진 프레임워크의 검사 규칙을 따른 보조 검사 두 가지.

- 근거 유효율: 모델이 낸 evidence가 입력 대화의 실제 글자를 가리키는가.
- 계획 위반 표현: B 후보가 완료를 주장하는 표현을 담고 있는가.

두 검사는 판정이나 정오(correct)에 영향을 주지 않고 보조 지표로만 보고한다.
규칙은 프레임워크의 것을 그대로 따른다. 프레임워크의 실제 함수와 같은 판정을 내리는지는
tests/test_evidence_and_claims.py가 대조해서 확인한다.
"""
import re

from .schema import is_int

# 프레임워크가 B 후보를 거절하는 정규식. 그대로 가져왔다.
COMPLETION_CLAIM = re.compile(r"(완료|저장|확정)(?:했|됐|되었|되셨|되었습니다)")
COMPLETE_ACTION = "COMPLETE"


def evidence_valid(evidence, context, must_be_last_turn):
    """근거 하나가 유효한지 본다. evidence는 스키마 검사를 통과한 dict여야 한다.

    ① turn_index가 context 범위 안, ② 그 턴이 user, ③ 0 <= start < end <= 글자 수,
    ④ content[start:end]가 text와 같음, ⑤ must_be_last_turn이면 마지막 턴(현재 발화)이어야 한다.
    """
    turn, start, end = evidence["turn_index"], evidence["start"], evidence["end"]
    if not (is_int(turn) and is_int(start) and is_int(end)):
        return False
    if not 0 <= turn < len(context):
        return False
    message = context[turn]
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, str) or not 0 <= start < end <= len(content):
        return False
    if content[start:end] != evidence["text"]:
        return False
    return not must_be_last_turn or turn == len(context) - 1


def evidence_items(obj):
    """출력의 모든 근거를 (evidence, 마지막 턴이어야 하는가) 쌍의 목록으로 돌려준다."""
    items = []
    for key in ("facts", "intents", "relations"):
        items += [(entry["evidence"], True) for entry in obj[key]]
    items += [(entry["evidence"], False) for entry in obj["unmapped_facts"]]
    return items


def evidence_report(obj, context):
    """(근거 항목 수, 유효한 항목 수, 모든 근거가 유효한가). obj는 스키마를 통과한 출력이다."""
    items = evidence_items(obj)
    valid = sum(1 for evidence, last in items if evidence_valid(evidence, context, last))
    return len(items), valid, valid == len(items)


def has_completion_claim(text, plan_action):
    """완료 단계가 아닌데 완료를 주장하는 표현이 있으면 True. 문자열이 아니거나 비어 있으면 False."""
    if not isinstance(text, str) or not text.strip():
        return False
    return plan_action != COMPLETE_ACTION and COMPLETION_CLAIM.search(text) is not None
