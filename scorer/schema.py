"""역할 A의 출력 형식: 스키마 검사, 값 규칙, 칸 표."""
import re

TOP_KEYS = ("facts", "intents", "relations", "unmapped_facts")
FACT_KEYS = {"item_id", "field", "value", "semantic_status", "precision", "evidence"}
INTENT_KEYS = {"item_id", "intent", "evidence"}
RELATION_KEYS = {"item_id", "relation", "correction_target", "evidence"}
UNMAPPED_KEYS = {"evidence", "scope"}
EVIDENCE_KEYS = {"turn_index", "start", "end", "text"}
TARGET_KEYS = {"item_id", "field", "turn_index"}

SEMANTIC_STATUS = ("ANSWERED", "MISSING", "AMBIGUOUS", "UNCERTAIN")
PRECISION = ("exact", "approximate", "range", "unspecified")
INTENTS = ("ANSWER", "ASK_PURPOSE", "REQUEST_SKIP", "REFUSED", "OFF_TOPIC")
RELATIONS = ("NEW_INFORMATION", "CORRECTION", "CONFLICT")
SCOPES = ("OUT_OF_SCHEMA",)


class FieldSpec:
    """칸 표의 칸 하나. kind는 boolean, integer, number, enum 중 하나."""

    def __init__(self, kind, minimum=None, maximum=None, exclusive_minimum=None, allowed=None):
        """칸 정의 하나를 만든다. 종류, 최소, 최대, 초과 최소(exclusive_minimum), 허용값."""
        self.kind = kind
        self.minimum = minimum
        self.maximum = maximum
        self.exclusive_minimum = exclusive_minimum
        self.allowed = tuple(allowed) if allowed is not None else None


def _build_table():
    """칸 표를 만든다. 문항 id마다 {칸 이름: FieldSpec}."""
    table = {}
    for n in range(1, 12):
        table["Q1.D%02d" % n] = {"diagnosed": FieldSpec("boolean"), "on_medication": FieldSpec("boolean")}
    for n in range(1, 6):
        table["Q2.D%02d" % n] = {"family_history": FieldSpec("boolean")}
    table["Q3"] = {"answer": FieldSpec("enum", allowed=("yes", "no", "unknown"))}
    for q in ("Q4", "Q5", "Q6"):
        table[q] = {"answer": FieldSpec("boolean")}
    for q in ("Q4_1", "Q5_1"):
        table[q] = {
            "status": FieldSpec("enum", allowed=("current", "former")),
            "total_years": FieldSpec("integer", minimum=0),
            "daily_count": FieldSpec("number", minimum=0),
            "years_since_quit": FieldSpec("number", minimum=0),
        }
    table["Q6_1"] = {"usage_days": FieldSpec("integer")}
    table["Q7"] = {
        "frequency": FieldSpec("number", exclusive_minimum=0),
        "unit": FieldSpec("enum", allowed=("week", "month", "year")),
        "does_not_drink": FieldSpec("boolean"),
    }
    for q in ("Q8_1", "Q9_1", "Q10"):
        table[q] = {"answer": FieldSpec("integer", minimum=0, maximum=7)}
    for q in ("Q8_2", "Q9_2"):
        table[q] = {
            "hours": FieldSpec("integer", minimum=0),
            "minutes": FieldSpec("integer", minimum=0, maximum=59),
        }
    return table


FIELD_TABLE = _build_table()
BEVERAGES = ("소주", "맥주", "양주", "막걸리", "와인")
BEVERAGE_UNITS = ("잔", "병", "캔", "cc")
_AMOUNT_FIELD = re.compile(r"^amounts\[(0|[1-9][0-9]*)\]\.(beverage|amount|unit)$")
_AMOUNT_SPECS = {
    "beverage": FieldSpec("enum", allowed=BEVERAGES),
    "amount": FieldSpec("number"),
    "unit": FieldSpec("enum", allowed=BEVERAGE_UNITS),
}
ITEM_IDS = frozenset(list(FIELD_TABLE) + ["Q7_1", "Q7_2"])


def field_spec(item_id, field):
    """(item_id, field)의 FieldSpec. 칸 표에 없는 칸이면 None."""
    if item_id in ("Q7_1", "Q7_2"):
        match = _AMOUNT_FIELD.match(field) if isinstance(field, str) else None
        return _AMOUNT_SPECS[match.group(2)] if match else None
    return FIELD_TABLE.get(item_id, {}).get(field)


def is_int(value):
    """JSON 정수인지 본다. bool은 정수로 보지 않는다."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value):
    """JSON 수(정수 또는 실수)인지 본다. bool은 수로 보지 않는다."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _evidence_errors(evidence, where):
    """evidence 객체의 키와 타입을 검사해 문제 목록을 돌려준다."""
    if not isinstance(evidence, dict) or set(evidence) != EVIDENCE_KEYS:
        return ["%s: evidence keys must be exactly %s" % (where, sorted(EVIDENCE_KEYS))]
    errors = []
    for key in ("turn_index", "start", "end"):
        if not is_int(evidence[key]):
            errors.append("%s: evidence.%s must be an integer" % (where, key))
    if not isinstance(evidence["text"], str):
        errors.append("%s: evidence.text must be a string" % where)
    return errors


def _keys_ok(item, keys):
    """dict이고 키 집합이 정확히 keys와 같은지 본다."""
    return isinstance(item, dict) and set(item) == keys


def schema_errors(obj):
    """읽은 객체의 스키마 검사. 빈 목록이면 schema_ok.

    키 이름이 정확한지, 값의 JSON 타입이 맞는지. 허용값(예: semantic_status가 네 가지 중 하나인지)과
    칸 규칙(값의 타입과 범위)은 값 규칙(value_errors)에서 본다.
    문제를 발견해도 멈추지 않고 모아서 목록으로 돌려주며, 항목 하나의 키가 틀리면
    그 항목의 나머지 검사만 건너뛴다(키가 없으면 값을 꺼낼 수 없음).
    """
    errors = []

    # 최상위: 키가 정확히 네 개여야 함.
    if not isinstance(obj, dict) or set(obj) != set(TOP_KEYS):
        return ["top-level keys must be exactly %s" % list(TOP_KEYS)]
    # 네 값이 모두 배열이어야 함.
    for key in TOP_KEYS:
        if not isinstance(obj[key], list):
            errors.append("%s must be a list" % key)
    if errors:
        return errors

    # facts: 키 6개가 정확히 있어야 함. item_id, field, semantic_status, precision은 문자열
    # value는 어떤 JSON이어도 통과시킨다(칸 종류에 따른 타입은 값 규칙에서 봄).
    for i, item in enumerate(obj["facts"]):
        where = "facts[%d]" % i
        if not _keys_ok(item, FACT_KEYS):
            errors.append("%s: keys must be exactly %s" % (where, sorted(FACT_KEYS)))
            continue
        for key in ("item_id", "field", "semantic_status", "precision"):
            if not isinstance(item[key], str):
                errors.append("%s.%s must be a string" % (where, key))
        errors += _evidence_errors(item["evidence"], where)

    # intents: 키 3개. item_id와 intent는 문자열.
    for i, item in enumerate(obj["intents"]):
        where = "intents[%d]" % i
        if not _keys_ok(item, INTENT_KEYS):
            errors.append("%s: keys must be exactly %s" % (where, sorted(INTENT_KEYS)))
            continue
        for key in ("item_id", "intent"):
            if not isinstance(item[key], str):
                errors.append("%s.%s must be a string" % (where, key))
        errors += _evidence_errors(item["evidence"], where)

    # relations: 키 4개. correction_target은 키가 반드시 있어야 하고, 값은 null이거나
    # 키 3개(item_id, field, turn_index)가 정확한 객체. 정답의 CORRECTION은 객체를 가짐.
    for i, item in enumerate(obj["relations"]):
        where = "relations[%d]" % i
        if not _keys_ok(item, RELATION_KEYS):
            errors.append("%s: keys must be exactly %s" % (where, sorted(RELATION_KEYS)))
            continue
        for key in ("item_id", "relation"):
            if not isinstance(item[key], str):
                errors.append("%s.%s must be a string" % (where, key))
        target = item["correction_target"]
        if target is not None:
            if not _keys_ok(target, TARGET_KEYS):
                errors.append("%s: correction_target keys must be exactly %s" % (where, sorted(TARGET_KEYS)))
            else:
                if not isinstance(target["item_id"], str) or not isinstance(target["field"], str):
                    errors.append("%s: correction_target item_id and field must be strings" % where)
                if not is_int(target["turn_index"]):
                    errors.append("%s: correction_target.turn_index must be an integer" % where)
        errors += _evidence_errors(item["evidence"], where)

    # unmapped_facts: 키 2개(evidence, scope). scope는 문자열(허용값 OUT_OF_SCHEMA는 값 규칙).
    for i, item in enumerate(obj["unmapped_facts"]):
        where = "unmapped_facts[%d]" % i
        if not _keys_ok(item, UNMAPPED_KEYS):
            errors.append("%s: keys must be exactly %s" % (where, sorted(UNMAPPED_KEYS)))
            continue
        if not isinstance(item["scope"], str):
            errors.append("%s.scope must be a string" % where)
        errors += _evidence_errors(item["evidence"], where)
    return errors


def _value_error(spec, value):
    """값이 칸 정의에 맞으면 None, 아니면 사유 문자열."""
    if spec.kind == "boolean":
        return None if isinstance(value, bool) else "must be a JSON boolean"
    if spec.kind == "integer":
        if not is_int(value):
            return "must be a JSON integer"
    elif spec.kind == "number":
        if not is_number(value):
            return "must be a JSON number"
    elif spec.kind == "enum":
        if not isinstance(value, str) or value not in spec.allowed:
            return "must be one of %s" % list(spec.allowed)
        return None
    if spec.minimum is not None and value < spec.minimum:
        return "below minimum %s" % spec.minimum
    if spec.maximum is not None and value > spec.maximum:
        return "above maximum %s" % spec.maximum
    if spec.exclusive_minimum is not None and value <= spec.exclusive_minimum:
        return "must be greater than %s" % spec.exclusive_minimum
    return None


def value_errors(obj, questionnaire_ids):
    """스키마 검사를 통과한 객체에 대한 값 규칙."""
    errors = []
    seen = set()
    for i, fact in enumerate(obj["facts"]):
        where = "facts[%d]" % i
        item_id, field = fact["item_id"], fact["field"]
        spec = None
        if item_id not in questionnaire_ids:
            errors.append("%s: item_id %r is not a question of the input" % (where, item_id))
        else:
            spec = field_spec(item_id, field)
            if spec is None:
                errors.append("%s: %r is not a field of %s" % (where, field, item_id))
        if (item_id, field) in seen:
            errors.append("%s: (%s, %s) appears more than once" % (where, item_id, field))
        seen.add((item_id, field))
        if fact["semantic_status"] not in SEMANTIC_STATUS:
            errors.append("%s: semantic_status %r not allowed" % (where, fact["semantic_status"]))
        if fact["precision"] not in PRECISION:
            errors.append("%s: precision %r not allowed" % (where, fact["precision"]))
        if fact["semantic_status"] == "ANSWERED":
            if spec is not None:
                problem = _value_error(spec, fact["value"])
                if problem:
                    errors.append("%s: value %s" % (where, problem))
            elif fact["value"] is None:
                errors.append("%s: ANSWERED needs a value" % where)
        elif fact["semantic_status"] in SEMANTIC_STATUS and fact["value"] is not None:
            errors.append("%s: value must be null unless ANSWERED" % where)
    for i, item in enumerate(obj["intents"]):
        if item["intent"] not in INTENTS:
            errors.append("intents[%d]: intent %r not allowed" % (i, item["intent"]))
    for i, item in enumerate(obj["relations"]):
        if item["relation"] not in RELATIONS:
            errors.append("relations[%d]: relation %r not allowed" % (i, item["relation"]))
    for i, item in enumerate(obj["unmapped_facts"]):
        if item["scope"] not in SCOPES:
            errors.append("unmapped_facts[%d]: scope %r not allowed" % (i, item["scope"]))
    return errors
