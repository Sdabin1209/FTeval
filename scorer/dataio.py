"""평가 데이터와 모델 출력 파일의 읽기와 검증."""
import hashlib
import json
from pathlib import Path

from . import schema
from .scoring_a import prepare_gold
from .strictjson import StrictJSONError, parse_strict_object


class InputError(Exception):
    """실행을 멈추게 하는 입력 문제."""


def read_jsonl(path):
    """JSONL 파일을 읽어 행 목록으로 돌려준다. 빈 줄은 무시하고, 읽을 수 없는 줄이 있으면 InputError."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise InputError("%s line %d is not valid JSON: %s" % (path, number, exc)) from exc
    return rows


class Dataset:
    """평가 데이터 두 파일을 한 번 읽은 것. 행은 파일 순서를 유지한다."""

    def __init__(self, data_dir):
        """data_dir의 extraction.jsonl(A)과 response.jsonl(B)을 읽고, id로 행을 찾는 색인을 만든다."""
        self.data_dir = Path(data_dir)
        self.extraction_path = self.data_dir / "extraction.jsonl"
        self.response_path = self.data_dir / "response.jsonl"
        self.extraction = read_jsonl(self.extraction_path)
        self.response = read_jsonl(self.response_path)
        self.ids = [r.get("id") for r in self.extraction]
        self._by_id_a = {r["id"]: r for r in self.extraction if isinstance(r, dict) and "id" in r}
        self._by_id_b = {r["id"]: r for r in self.response if isinstance(r, dict) and "id" in r}

    def row_a(self, case_id):
        """id에 해당하는 A용(extraction.jsonl) 행."""
        return self._by_id_a[case_id]

    def row_b(self, case_id):
        """id에 해당하는 B용(response.jsonl) 행."""
        return self._by_id_b[case_id]

    @staticmethod
    def user_json(row):
        """행의 사용자 메시지 content를 JSON으로 풀어 돌려준다. context, questionnaire, plan 등이 들어 있다."""
        return json.loads(row["prompt"][1]["content"])

    def questionnaire_ids(self, case_id):
        """사례의 입력에 들어 있는 문항 id 집합(31개). 모델 출력의 item_id가 유효한지 검사할 때 쓴다."""
        return frozenset(self.user_json(self.row_a(case_id))["questionnaire"])

    def gold_a(self, case_id):
        """A의 정답을 채점용으로 준비해 돌려준다(prepare_gold의 결과)."""
        return prepare_gold(json.loads(self.row_a(case_id)["completion"][0]["content"]))

    def reference(self, case_id):
        """B의 정답(참조) 문장."""
        return self.row_b(case_id)["completion"][0]["content"]

    def category(self, case_id):
        """사례 유형(category). 유형별 값 정확도 보고에 쓴다."""
        return self.row_a(case_id).get("category")

    def plan_action(self, case_id):
        """B 입력의 계획(plan)에 적힌 행동(NEXT, CLARIFY 등)."""
        return self.user_json(self.row_b(case_id))["plan"].get("action")

    def references(self):
        """모든 사례의 참조 문장을 파일 순서대로 돌려준다."""
        return [self.reference(i) for i in self.ids]

    def system_message(self, which):
        """A 또는 B 파일의 고정 지시문(시스템 메시지). prompt_hash 계산에 쓴다."""
        rows = self.extraction if which == "A" else self.response
        return rows[0]["prompt"][0]["content"]

    def dataset_hash(self, scope):
        """데이터 파일 바이트열의 SHA-256. scope는 both, A, B 중 하나."""
        digest = hashlib.sha256()
        if scope in ("both", "A"):
            digest.update(self.extraction_path.read_bytes())
        if scope in ("both", "B"):
            digest.update(self.response_path.read_bytes())
        return digest.hexdigest()

    def prompt_hash(self):
        """A와 B의 고정 지시문을 이어 붙인 문자열의 SHA-256. 실행 기록에 남겨 같은 입력으로 돌렸는지 확인한다."""
        text = (self.system_message("A") + "\n---\n" + self.system_message("B")).replace("\r\n", "\n")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_dataset(data):
    """평가 데이터 두 파일의 입력 검증. 문제 목록을 돌려준다(비어 있으면 이상 없음)."""
    problems = []

    # 각 파일에서 id가 모든 행에 있고 서로 겹치지 않는다.
    for name, rows in (("extraction.jsonl", data.extraction), ("response.jsonl", data.response)):
        ids = [r.get("id") if isinstance(r, dict) else None for r in rows]
        if len(set(map(str, ids))) != len(ids) or any(i is None for i in ids):
            problems.append("%s: ids are missing or not unique" % name)

    # 두 파일이 같은 사례(id) 집합을 가진다.
    ids_a = {r.get("id") for r in data.extraction}
    ids_b = {r.get("id") for r in data.response}
    if ids_a != ids_b:
        problems.append("the two files do not have the same set of ids")

    # 두 파일에 같은 검사를 적용한다.
    for name, rows in (("extraction.jsonl", data.extraction), ("response.jsonl", data.response)):
        for r in rows:
            where = "%s %s" % (name, r.get("id"))
            prompt = r.get("prompt")
            completion = r.get("completion")
            # prompt는 [system 메시지, user 메시지] 두 개여야 한다. 모양이 틀리면
            # 아래 검사를 할 수 없으므로 이 행은 여기서 건너뛴다(continue).
            if (not isinstance(prompt, list) or len(prompt) != 2
                    or [m.get("role") for m in prompt] != ["system", "user"]):
                problems.append("%s: prompt must be [system, user]" % where)
                continue
            # completion은 assistant 메시지 1개여야 한다.
            if (not isinstance(completion, list) or len(completion) != 1
                    or completion[0].get("role") != "assistant"):
                problems.append("%s: completion must be one assistant message" % where)
                continue
            # user 메시지의 content가 JSON으로 읽혀야 한다.
            try:
                json.loads(prompt[1]["content"])
            except (ValueError, TypeError):
                problems.append("%s: user content is not JSON" % where)
        # (행마다가 아니라 파일 전체에 대해) system 메시지가 모든 행에서 같아야 한다.
        # 종류가 둘 이상이면 prompt_hash를 하나로 정할 수 없다.
        systems = {r["prompt"][0]["content"] for r in rows if isinstance(r.get("prompt"), list) and r["prompt"]}
        if len(systems) > 1:
            problems.append("%s: the system message differs between rows" % name)

    # 위의 모양 검사에서 문제가 났다면, 아래 검사는 그 모양을 전제로 하므로 여기서 멈춘다.
    if problems:
        return problems

    # response.jsonl: 사용자 메시지에 plan이 있고 정답 문장이 비어 있지 않다.
    for r in data.response:
        user = json.loads(r["prompt"][1]["content"])
        if "plan" not in user:
            problems.append("response.jsonl %s: user content has no plan" % r["id"])
        content = r["completion"][0].get("content")
        if not isinstance(content, str) or not content.strip():
            problems.append("response.jsonl %s: completion is empty" % r["id"])

    # extraction.jsonl: 정답 자체가 모델에게 요구하는 출력 규칙(엄격한 JSON, 스키마,
    # 값 규칙)을 통과한다. 정답이 규칙에 어긋나면 채점 기준이 틀린 것이다.
    for r in data.extraction:
        case_id = r["id"]
        try:
            obj = parse_strict_object(r["completion"][0]["content"])
        except StrictJSONError as exc:
            problems.append("extraction.jsonl %s: gold is not valid JSON: %s" % (case_id, exc))
            continue
        errors = schema.schema_errors(obj)
        if not errors:   # 값 규칙은 스키마를 통과한 객체에만 적용할 수 있다
            errors = schema.value_errors(obj, data.questionnaire_ids(case_id))
        for message in errors:
            problems.append("extraction.jsonl %s: gold breaks the output rules: %s" % (case_id, message))
    return problems


def read_outputs(path, key, valid_ids):
    """모델 출력 파일 하나를 읽는다.

    (values, warnings)를 돌려준다. values는 평가 데이터에 있는 id마다 `key`의 값을 담는다.
    파일에 없는 id는 dict에 넣지 않으며, 그런 사례는 결측으로 채점된다.
    """
    path = Path(path)
    values = {}
    warnings = []
    seen = set()
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():  # 빈 줄 스킵
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:  # JSON 형식 오류
                raise InputError("%s line %d is not valid JSON: %s" % (path, number, exc)) from exc
            if not isinstance(row, dict):  # JSON 객체 아님
                raise InputError("%s line %d is not a JSON object" % (path, number))
            if "id" not in row or key not in row:  # id 혹은 key 없음
                raise InputError("%s line %d needs the keys id and %s" % (path, number, key))
            marker = json.dumps(row["id"], sort_keys=True, ensure_ascii=False)
            if marker in seen:  # id 중복
                raise InputError("%s line %d: id %s appears more than once" % (path, number, marker))
            seen.add(marker)
            if not isinstance(row["id"], str) or row["id"] not in valid_ids:  # id가 평가 사례에 없음
                warnings.append("%s line %d: id %s is not an evaluation case and was ignored"
                                % (path.name, number, marker))
                continue
            values[row["id"]] = row[key]
    return values, warnings
