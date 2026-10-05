"""채점기가 읽는 모양의 모델 출력 파일을 만들어 파이프라인을 시험해 볼 때 쓴다.

파일은 정답 라벨에 정해진 방식의 손상을 더해 만든다. 모델은 아니고 모델을 흉내 낸 것이다.
이 모듈을 import만 해서는 아무것도 쓰지 않는다. 테스트는 임시 폴더를 넘겨 write_mock_run을
호출한다. runs/에 쓰는 것은 모델이 낸 출력 모양을 시험해 볼 때뿐이다.
"""
import json
import random
from pathlib import Path

CORRUPTIONS_A = ("drop_fact", "wrong_value", "not_json", "code_block", "extra_refused", "empty_facts",
                 "bad_evidence")


def _damage_a(gold, kind, rng):
    text = json.dumps(gold, ensure_ascii=False)
    if kind == "not_json":
        return "I could not extract anything."
    if kind == "code_block":
        return "```json\n" + text + "\n```"
    out = json.loads(text)
    if kind == "drop_fact" and out["facts"]:
        out["facts"].pop(rng.randrange(len(out["facts"])))
    elif kind == "wrong_value" and out["facts"]:
        fact = out["facts"][0]
        value = fact["value"]
        fact["value"] = (not value) if isinstance(value, bool) else (value + 1 if isinstance(value, int) else value)
    elif kind == "extra_refused" and out["facts"]:
        out["intents"].append({"item_id": out["facts"][0]["item_id"], "intent": "REFUSED",
                               "evidence": out["facts"][0]["evidence"]})
    elif kind == "empty_facts":
        out["facts"] = []
    elif kind == "bad_evidence" and out["facts"]:
        out["facts"][0]["evidence"]["text"] += "x"      # 값은 맞고 근거 글자만 틀리다
    return json.dumps(out, ensure_ascii=False)


def mock_a(data, accuracy, seed):
    rng = random.Random(seed)
    rows = []
    for case_id in data.ids:
        gold = json.loads(data.row_a(case_id)["completion"][0]["content"])
        if rng.random() < accuracy:
            raw = json.dumps(gold, ensure_ascii=False)
        else:
            raw = _damage_a(gold, rng.choice(CORRUPTIONS_A), rng)
        rows.append({"id": case_id, "raw_output": raw})
    return rows


def mock_b(data, quality, seed):
    rng = random.Random(seed)
    rows = []
    refs = data.references()
    for case_id in data.ids:
        ref = data.reference(case_id)
        if rng.random() < quality:
            candidate = ref
        else:
            choice = rng.choice(("drop_last", "other", "empty", "claim"))
            if choice == "drop_last":
                candidate = " ".join(ref.split()[:-1])
            elif choice == "other":
                candidate = rng.choice(refs)
            elif choice == "claim":
                candidate = ref + " 입력하신 내용이 저장되었습니다."     # 완료를 주장하는 표현을 덧붙인다
            else:
                candidate = ""
        rows.append({"id": case_id, "candidate": candidate})
    return rows


def write_mock_run(runs_dir, run_id, data, quality=1.0, seed=0, roles=("A", "B")):
    """runs_dir/<run_id>/m1v_outputs.jsonl과 m2p_outputs.jsonl을 쓰고 그 폴더를 돌려준다."""
    folder = Path(runs_dir) / run_id
    folder.mkdir(parents=True, exist_ok=True)
    if "A" in roles:
        _write(folder / "m1v_outputs.jsonl", mock_a(data, quality, seed))
    if "B" in roles:
        _write(folder / "m2p_outputs.jsonl", mock_b(data, quality, seed + 1))
    return folder


def _write(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
