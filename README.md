# FineTuningEval

역할 A(항목 값 추출)와 역할 B(계획 → 한국어 문장) 모델 출력을 채점하고, base와 파인튜닝 결과를 비교하는 도구입니다.

## 준비

```
pip install -r requirements.txt
```

평가 데이터(`extraction.jsonl`, `response.jsonl`)는 저장소에 없습니다.
`dataset/labeled_questionnaire_100_handoff/data/`에 직접 두세요. (`config.json`의 `data_dir`)

## 명령줄

모델 출력을 `runs/<이름>/m1v_outputs.jsonl`(A), `m2p_outputs.jsonl`(B)로 두세요.
이름은 `base` 또는 `ft_seed<시드>`(예: `ft_seed42`)여야 `compare`가 읽습니다.
한 줄에 사례 하나이고, 키는 아래와 같습니다. 모델이 낸 문자열은 고치지 말고 그대로 넣으세요.

```
{"id": "labeled100_001", "raw_output": "<모델이 낸 JSON 문자열>"}    ← m1v_outputs.jsonl (A)
{"id": "labeled100_001", "candidate": "<응답 문장>"}                 ← m2p_outputs.jsonl (B)
```

```
python -m scorer validate           # 평가 데이터 검사
python -m scorer score base         # runs/base 채점 → results/base/
python -m scorer score ft_seed42
python -m scorer compare            # base와 ft_seed* 비교 → results/comparison.json
python -m scorer report             # → results/report.md
python -m scorer selftest           # 검증 사례 실행 (평가 데이터 필요)
```

`score`는 `--model-id`, `--decoding <json 파일>`로 실행 정보를 함께 기록할 수 있습니다.
`--config`는 명령 앞에 둡니다: `python -m scorer --config config.json score base`.

## SDK

파이썬 코드에서 직접 부릅니다. 모델 출력을 dict로 넘기면 파일을 만들지 않고 결과를 돌려줍니다.

```python
from scorer import Evaluator

ev = Evaluator("config.json")        # 평가 데이터는 한 번만 읽고 재사용합니다

# 1) 모델에 넣을 입력 (바꾸지 말고 그대로 넣으세요)
ids = ev.data.ids                    # 사례 id 목록
ev.data.row_a(ids[0])["prompt"]      # 역할 A 입력 [system, user]
ev.data.row_b(ids[0])["prompt"]      # 역할 B 입력
#    → 모델이 낸 문자열을 {사례 id: 출력} dict로 모아 outputs_a, outputs_b로 만듭니다.

# 2) 채점
base = ev.score("base", outputs_a, outputs_b, model_id="...", decoding={...})
ft = ev.score("ft_seed42", outputs_a, outputs_b, model_id="...", decoding={...})

# 3) 결과
base.a_summary["value_accuracy"]     # A 값 정확도
base.b_summary["fm_mean"]            # B 문장 자연스러움(FM) 평균
comparison = ev.compare([base, ft])  # 신뢰구간과 판정
print(ev.report([base, ft], comparison))   # report.md와 같은 글
ev.save(base)                        # 파일로 남기고 싶을 때만 results/에 씁니다
```

- `outputs_a`: 사례 id → 모델이 낸 JSON 문자열, `outputs_b`: 사례 id → 응답 문장입니다. 평가하지 않는 역할은 `None`으로 둡니다. 빠진 사례는 결측(오답)으로 세고 `result.warnings`에 경고가 담깁니다.
- `decoding`: 모델을 돌릴 때 쓴 설정(최대 토큰 수, JSON 모드 등)을 dict로 넘깁니다. 채점기가 값을 정하지 않고 받은 값을 그대로 기록하며, 실행끼리 다르면 리포트에 경고합니다.
- `compare`/`report`에 쓰려면 실행 이름이 `base` 또는 `ft_seed<시드>`여야 합니다.
- `ev.score`가 돌려주는 `RunResult`에는 `a_scores`, `a_summary`, `b_scores`, `b_summary`, `meta`, `warnings`가 있습니다.
- `ev.score_folder("base")`는 명령줄 `score`와 같습니다(`runs/`를 읽어 `results/`에 씀).
- 한 번만 채점하면 `scorer.evaluate(outputs_a, outputs_b)`로 충분합니다.
- 설치하는 패키지가 아닙니다. 이 폴더에서 실행하거나, 다른 폴더의 코드에서는 먼저 `sys.path.insert(0, "<이 폴더 경로>")`를 하고 설정 파일은 전체 경로로 넘기세요: `Evaluator("<이 폴더 경로>/config.json")`.

## 참고

- `runs/`는 모델 출력 파일을 두는 곳이고, `results/`는 채점기가 만드는 산출물 폴더입니다. `results/` 안의 파일은 직접 고치지 마세요.
- 문장 점수용 코퍼스는 명령줄로 채점하면 `results/corpus.txt`에 처음 한 번 만들어지고, 이후에는 그 파일을 읽습니다(정답 문장과 같은 줄은 뺍니다). SDK는 파일이 없으면 만들어 쓰지 않고 메모리에서만 만듭니다.
- 코퍼스가 `corpus_min_sentences`보다 작으면 B 요약에 `corpus_below_minimum: True`가 뜹니다.
