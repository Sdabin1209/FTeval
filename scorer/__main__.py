"""명령줄: python -m scorer <명령> [--config config.json]."""
import argparse
import sys
import unittest
from pathlib import Path

from . import pipeline, report, scoring_b
from .dataio import InputError


def _config(args):
    """명령줄 인자의 --config 경로로 Config 객체를 만든다."""
    return pipeline.Config(args.config)


def cmd_validate(args):
    """validate 명령. 평가 데이터 두 파일이 서로 맞는지, 정답이 출력 규칙을 통과하는지 검사해 결과를 출력한다. 문제가 있으면 종료 코드 1."""
    config = _config(args)
    data, problems = pipeline.check_inputs(config)
    if problems:
        print("input check FAILED (%d problems):" % len(problems))
        for p in problems[:50]:
            print("  -", p)
        return 1
    print("input check passed: %d cases, both files consistent, gold passes the output rules" % len(data.ids))
    return 0


def cmd_build_corpus(args):
    """build-corpus 명령. 데이터의 챗봇 문장에서 참조 문장을 뺀 코퍼스를 만들어 설정의 코퍼스 경로에 쓴다."""
    config = _config(args)
    data, problems = pipeline.check_inputs(config)
    if problems:
        print("input check failed; run validate first")
        return 1
    sentences = scoring_b.build_corpus(data)
    scoring_b.write_corpus(sentences, config.corpus_path)
    print("wrote %d corpus sentences to %s" % (len(sentences), config.corpus_path))
    return 0


def cmd_score(args):
    """score 명령. runs/<실행 id>의 출력 파일을 채점해 results/<실행 id>/에 점수, 요약, 실행 기록을 쓴다."""
    config = _config(args)
    data, problems = pipeline.check_inputs(config)
    if problems:
        print("input check failed; run validate first")
        return 1
    decoding = None
    if args.decoding:
        try:
            decoding = pipeline.read_json(args.decoding)
        except (OSError, ValueError) as exc:
            raise InputError("cannot read the decoding settings file %s: %s" % (args.decoding, exc)) from exc
    info = pipeline.score_run_folder(config, data, args.run, args.model_id, decoding)
    print("scored %s -> %s" % (args.run, config.results_dir / args.run))
    for name in info["files"]:
        print("  wrote", name)
    for w in info["warnings"]:
        print("  warning:", w)
    return 0


def cmd_compare(args):
    """compare 명령. results/ 아래의 base와 ft 실행을 비교해 results/comparison.json에 쓴다."""
    config = _config(args)
    comparison = pipeline.compare_runs(config)
    pipeline.write_json(config.results_dir / "comparison.json", comparison)
    print("A: mode %s, overall %s | B: mode %s" % (
        comparison["a"]["mode"], comparison["a"]["overall"], comparison["b"]["mode"]))
    print("wrote", config.results_dir / "comparison.json")
    return 0


def cmd_report(args):
    """report 명령. 비교 결과(comparison.json, 없으면 새로 계산)로 results/report.md를 쓴다."""
    config = _config(args)
    data, problems = pipeline.check_inputs(config)
    if problems:
        print("input check failed; run validate first")
        return 1
    path = config.results_dir / "comparison.json"
    comparison = pipeline.read_json(path) if path.exists() else pipeline.compare_runs(config)
    print("wrote", report.write_report(config, comparison, data))
    return 0


def cmd_selftest(args):
    """selftest 명령. tests/ 의 검증 사례를 모두 실행한다. 전부 통과해야 종료 코드 0."""
    here = Path(__file__).resolve().parent.parent
    suite = unittest.defaultTestLoader.discover(str(here / "tests"), top_level_dir=str(here))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    """명령줄을 해석해 명령을 실행한다. 입력 문제(InputError)는 메시지를 출력하고 종료 코드 2로 끝낸다."""
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(prog="scorer", description=__doc__)
    parser.add_argument("--config", default=str(root / "config.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="check the evaluation data").set_defaults(func=cmd_validate)
    sub.add_parser("build-corpus", help="write the corpus file (3.4)").set_defaults(func=cmd_build_corpus)
    p = sub.add_parser("score", help="score one run folder under runs/")
    p.add_argument("run", help="run id, for example base or ft_seed42")
    p.add_argument("--model-id", default=None, help="model identifier stored in run_meta.json")
    p.add_argument("--decoding", default=None,
                   help="JSON file with the decoding settings the model run used (recorded as given)")
    p.set_defaults(func=cmd_score)
    sub.add_parser("compare", help="compare base and ft runs found under results/").set_defaults(func=cmd_compare)
    sub.add_parser("report", help="write results/report.md").set_defaults(func=cmd_report)
    sub.add_parser("selftest", help="run the verification cases (tests/)").set_defaults(func=cmd_selftest)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except InputError as exc:
        print("input error:", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
