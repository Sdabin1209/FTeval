"""실행 결과를 읽고 쓰는 저장소 두 가지: 디스크(results/ 폴더)와 메모리.

채점, 비교, 리포트 계산은 저장소가 무엇인지 모르고 같은 코드로 돌아간다. 그래서 파일로 쓰는
방식(명령줄)과 메모리에서 바로 부르는 방식(SDK)의 결과가 어긋나지 않는다. 한 실행의 결과는
파일 이름을 키로 하는 dict(artifacts)이다. 이름이 .jsonl로 끝나면 행 목록, 아니면 JSON 객체이다.
"""
import copy
import json
from pathlib import Path


def write_json(path, obj):
    """dict를 읽기 좋은 JSON 파일로 쓴다(한글 그대로, 폴더가 없으면 만든다)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    """행 목록을 한 줄에 하나씩 JSONL 파일로 쓴다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path):
    """JSON 파일을 읽어 돌려준다."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_rows(path):
    """JSONL 파일을 읽어 행 목록으로 돌려준다."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class DiskStore:
    """results/<실행 id>/<파일 이름> 형태의 폴더 저장소."""

    def __init__(self, root):
        """root는 results 폴더이다. 아직 없어도 된다(쓸 때 만든다)."""
        self.root = Path(root)

    def runs(self):
        """저장된 실행 id 목록(폴더 이름)을 정렬해 돌려준다."""
        if not self.root.is_dir():
            return []
        return sorted(child.name for child in self.root.iterdir() if child.is_dir())

    def has(self, run, name):
        """그 실행에 해당 파일이 있는지."""
        return (self.root / run / name).exists()

    def read_json(self, run, name):
        """실행의 JSON 파일 하나를 읽는다."""
        return read_json(self.root / run / name)

    def read_rows(self, run, name):
        """실행의 JSONL 파일 하나를 읽는다."""
        return read_rows(self.root / run / name)

    def write(self, run, artifacts):
        """artifacts(파일 이름 -> 내용)를 results/<run>/ 아래에 쓰고 쓴 파일 이름 목록을 돌려준다."""
        for name, content in artifacts.items():
            if name.endswith(".jsonl"):
                write_jsonl(self.root / run / name, content)
            else:
                write_json(self.root / run / name, content)
        return list(artifacts)


class MemoryStore:
    """파일을 만들지 않는 저장소. 내용은 JSON을 한 번 거쳐 저장해서 디스크에 쓰고 읽은 것과 같은 모양이다."""

    def __init__(self):
        """빈 저장소를 만든다."""
        self._runs = {}

    def runs(self):
        """저장된 실행 id 목록을 정렬해 돌려준다."""
        return sorted(self._runs)

    def has(self, run, name):
        """그 실행에 해당 이름의 결과가 있는지."""
        return name in self._runs.get(run, {})

    def read_json(self, run, name):
        """실행의 JSON 결과 하나를 복사해서 돌려준다."""
        return copy.deepcopy(self._runs[run][name])

    def read_rows(self, run, name):
        """실행의 행 목록 하나를 복사해서 돌려준다."""
        return copy.deepcopy(self._runs[run][name])

    def write(self, run, artifacts):
        """artifacts를 저장하고(같은 실행 id는 덮어쓴다) 저장한 이름 목록을 돌려준다."""
        self._runs[run] = {name: json.loads(json.dumps(content, ensure_ascii=False))
                           for name, content in artifacts.items()}
        return list(artifacts)
