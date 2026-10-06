"""파이썬에서 바로 부르는 SDK.

명령줄과 같은 계산을 쓰므로 두 방식의 결과는 같다. 차이는 입출력뿐이다. SDK는 모델 출력을
메모리의 dict로 받아 파일을 만들지 않고 결과를 돌려주며, 파일로 남기고 싶을 때만 save를 부른다.

    from scorer import Evaluator
    ev = Evaluator("config.json")
    base = ev.score("base", outputs_a={id: raw_output, ...}, outputs_b={id: candidate, ...},
                    model_id="...", decoding={"max_new_tokens_m1v": 1024})
    ft = ev.score("ft_seed42", outputs_a=..., outputs_b=..., decoding=...)
    comparison = ev.compare([base, ft])        # CI와 판정을 담은 dict
    markdown = ev.report([base, ft])           # report.md와 같은 글
"""
from pathlib import Path

from . import pipeline, scoring_b
from .dataio import InputError
from .pipeline import BASE, FT_PATTERN, Config, DiskStore, MemoryStore
from .report import build_report, write_report


class RunResult:
    """한 실행의 채점 결과. 파일 이름을 키로 하는 dict(artifacts)와 경고를 담는다."""

    def __init__(self, run_id, artifacts, warnings):
        """run_id, 결과 파일 dict, 경고 목록을 보관한다."""
        self.run_id = run_id
        self.artifacts = artifacts
        self.warnings = warnings

    @property
    def meta(self):
        """실행 기록(run_meta.json 내용): 모델 ID, 디코딩 설정, 해시, 경고 등."""
        return self.artifacts[pipeline.RUN_META]

    @property
    def a_scores(self):
        """A의 사례별 점수 행(m1v_scores.jsonl 내용). A를 채점하지 않았으면 None."""
        return self.artifacts.get(pipeline.A_SCORES)

    @property
    def a_summary(self):
        """A의 요약(m1v_summary.json 내용): 값 정확도, F1-OOS, 보조 지표, 진단. A를 채점하지 않았으면 None."""
        return self.artifacts.get(pipeline.A_SUMMARY)

    @property
    def b_scores(self):
        """B의 사례별 점수 행(m2p_scores.jsonl 내용). B를 채점하지 않았으면 None."""
        return self.artifacts.get(pipeline.B_SCORES)

    @property
    def b_summary(self):
        """B의 요약(m2p_summary.json 내용): FM·AM 평균과 분포 등. B를 채점하지 않았으면 None."""
        return self.artifacts.get(pipeline.B_SUMMARY)


class Evaluator:
    """평가 데이터와 설정을 한 번 읽어 두고, 실행을 여러 번 채점하고 비교하는 객체.

    평가 데이터 읽기와 문장 점수용 언어모델 만들기는 처음 한 번만 하고 재사용한다.
    """

    def __init__(self, config=None, *, check=True):
        """config는 config.json 경로, Config 객체, 또는 None(현재 폴더의 config.json)이다.

        check가 True이면 평가 데이터 검증을 하고 문제가 있으면 InputError를 낸다.
        """
        if config is None:
            config = Path("config.json")
        self.config = config if isinstance(config, Config) else Config(config)
        self.data = pipeline.Dataset(self.config.data_dir)
        if check:
            problems = pipeline.validate_dataset(self.data)
            if problems:
                raise InputError("evaluation data check failed (%d problems), first: %s" % (len(problems), problems[0]))
        self._memory = MemoryStore()
        self._sentence = None

    def _sentence_scorer(self):
        """설정에 맞는 문장 점수기(ngram이면 코퍼스로 만든 bigram과 LSI, neural이면 사전학습 모델)를 만들어 재사용한다. 파일은 만들지 않는다. (scorer, 경고)를 돌려준다."""
        if self._sentence is None:
            scorer, warnings = pipeline.make_sentence_scorer(self.config, self.data, write=False)
            self._sentence = (scorer, warnings)
        return self._sentence

    def score(self, run_id, outputs_a=None, outputs_b=None, *, model_id=None, decoding=None):
        """메모리의 모델 출력을 채점한다. 파일을 만들지 않고 RunResult를 돌려준다.

        run_id는 'base' 또는 'ft_seed<시드>' 형식이어야 비교에 쓸 수 있다. outputs_a는 사례 id에서
        모델이 낸 JSON 문자열로의 dict(역할 A), outputs_b는 사례 id에서 응답 문장으로의 dict(역할 B)이다.
        평가하지 않는 역할은 None으로 둔다. 빠진 id는 결측으로 센다. decoding은 모델을 돌린 쪽이 쓴
        디코딩 설정(dict)이고, 우리가 정하지 않고 받은 값을 그대로 기록한다.
        """
        ids = set(self.data.ids)
        input_warnings, a_values, b_values, scorer = [], None, None, None
        if outputs_a is not None:
            a_values, w = pipeline.normalize_outputs(outputs_a, ids, "m1v_outputs")
            input_warnings += w
        if outputs_b is not None:
            b_values, w = pipeline.normalize_outputs(outputs_b, ids, "m2p_outputs")
            input_warnings += w
            scorer, corpus_warnings = self._sentence_scorer()
            input_warnings += corpus_warnings
        artifacts, warnings = pipeline.score_outputs(
            self.config, self.data, run_id, a_values, b_values, model_id=model_id, decoding=decoding,
            sentence_scorer=scorer, input_warnings=input_warnings)
        self._memory.write(run_id, artifacts)
        return RunResult(run_id, artifacts, warnings)

    def score_folder(self, run_id, *, model_id=None, decoding=None):
        """명령줄의 score와 같다. runs/<run_id>/의 출력 파일을 읽어 results/에 쓰고 RunResult를 돌려준다."""
        info = pipeline.score_run_folder(self.config, self.data, run_id, model_id, decoding)
        store = DiskStore(self.config.results_dir)
        artifacts = {name: (store.read_rows(run_id, name) if name.endswith(".jsonl")
                            else store.read_json(run_id, name)) for name in info["files"]}
        self._memory.write(run_id, artifacts)
        return RunResult(run_id, artifacts, info["warnings"])

    def save(self, result):
        """RunResult를 results/<run_id>/ 아래 파일로 쓴다. 쓴 파일 이름 목록을 돌려준다."""
        return DiskStore(self.config.results_dir).write(result.run_id, result.artifacts)

    def _store_of(self, results):
        """results(RunResult 목록)로 메모리 저장소를 만든다. None이면 이 객체가 채점한 모든 실행을 쓴다."""
        if results is None:
            return self._memory
        store = MemoryStore()
        for result in results:
            if result.run_id != BASE and not FT_PATTERN.match(result.run_id):
                raise InputError("run id %r is neither 'base' nor 'ft_seed<N>'" % result.run_id)
            store.write(result.run_id, result.artifacts)
        return store

    def compare(self, results=None):
        """RunResult들을 비교해 CI와 판정을 담은 dict를 돌려준다(comparison.json과 같은 내용)."""
        return pipeline.compare_runs(self.config, self._store_of(results))

    def report(self, results=None, comparison=None):
        """report.md와 같은 글(문자열)을 돌려준다. 파일을 만들지 않는다."""
        store = self._store_of(results)
        comparison = comparison if comparison is not None else pipeline.compare_runs(self.config, store)
        return build_report(self.config, comparison, self.data, store)

    def compare_saved(self):
        """results/ 폴더에 저장된 실행들을 비교한다(명령줄의 compare와 같다)."""
        return pipeline.compare_runs(self.config)

    def write_saved_report(self):
        """results/ 폴더에 저장된 실행들로 results/report.md를 쓰고 그 경로를 돌려준다."""
        return write_report(self.config, pipeline.compare_runs(self.config), self.data)


def evaluate(outputs_a=None, outputs_b=None, *, run_id=BASE, config=None, model_id=None, decoding=None):
    """한 실행을 한 번에 채점하는 간편 함수. 비교가 필요하면 Evaluator를 직접 쓴다."""
    return Evaluator(config).score(run_id, outputs_a, outputs_b, model_id=model_id, decoding=decoding)
