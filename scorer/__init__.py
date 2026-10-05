"""평가 파이프라인 채점기.

명령줄은 `python -m scorer`, 파이썬에서 부르려면 `from scorer import Evaluator`를 쓴다.
"""
from .api import Evaluator, RunResult, evaluate

__all__ = ["Evaluator", "RunResult", "evaluate"]
