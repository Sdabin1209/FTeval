"""테스트 공용 도구. 실제 데이터셋은 읽기만 하고, 테스트는 임시 폴더에만 쓴다."""
import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from scorer import scoring_a
from scorer.dataio import Dataset
from scorer.pipeline import Config

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"


def fingerprint(folder):
    """folder 아래 모든 파일의 (상대 경로, 크기, 수정 시각, sha256). 폴더가 없으면 None.

    "폴더가 비어 있다"와 "폴더가 없다"를 구분하려고 없을 때는 빈 목록이 아니라 None을 돌려준다.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return None
    out = []
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            out.append((str(path.relative_to(folder)), path.stat().st_size, path.stat().st_mtime_ns,
                        hashlib.sha256(path.read_bytes()).hexdigest()))
    return out


def project_state():
    """프로젝트의 runs/와 results/가 지금 어떤 상태인지 찍는다. 테스트 전후로 비교하는 데 쓴다.

    테스트가 이 두 폴더에 아무것도 쓰지 않았는지는 "비어 있다/없다"가 아니라 "전후가 같다"로 확인한다.
    실제 모델 출력이나 실제 채점 결과가 들어 있어도 테스트가 틀리지 않게 하기 위해서이다.
    """
    return {name: fingerprint(ROOT / name) for name in ("runs", "results")}


@lru_cache(maxsize=1)
def config():
    return Config(CONFIG_PATH)


@lru_cache(maxsize=1)
def dataset():
    return Dataset(config().data_dir)


def gold_json(case_id):
    """사례의 정답 completion을 새 dict로 돌려준다(수정해도 안전)."""
    return copy.deepcopy(json.loads(dataset().row_a(case_id)["completion"][0]["content"]))


def score_a(case_id, output):
    """사례에 대한 모델 출력(dict 또는 원문 문자열)을 채점한다."""
    raw = json.dumps(output, ensure_ascii=False) if isinstance(output, dict) else output
    data = dataset()
    context = data.user_json(data.row_a(case_id))["context"]
    return scoring_a.score_case(raw, data.gold_a(case_id), data.questionnaire_ids(case_id), context)
