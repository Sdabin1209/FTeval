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

## 문장 점수 방식 (역할 B의 FM, AM)

기본(`config.json`)은 `ngram`입니다. 코퍼스로 bigram 언어모델(FM)과 LSI(AM)를 만들고 numpy만 씁니다.
코퍼스가 작으면 점수가 코퍼스 어절과 얼마나 겹치는지에 좌우되므로, 사전학습 모델을 쓰는 `neural` 방식도 있습니다.
FM은 언어모델의 토큰당 평균 로그확률, AM은 문장 임베딩의 코사인 유사도로 계산하고, 코퍼스는 쓰지 않습니다.

```
pip install -r requirements-neural.txt
python -m scorer --config config.neural.json score base
```

- `config.neural.json`은 `sentence_scorer`의 모델(FM `Qwen/Qwen2.5-1.5B`, AM `BAAI/bge-m3`)과 결과 폴더(`results_neural/`)를 바꾼 설정입니다. 모델은 처음 실행할 때 내려받습니다(인터넷, 약 5.4GB 디스크 필요).
- 두 모델 모두 `revision`(모델 버전)을 고정해 두었고, 정밀도는 `float32`로 명시했습니다. 모델 이름은 `lm.model`, `embedder.model`에서 바꿉니다. 바꿀 때는 `revision`을 비워 두지 말고 그 모델의 버전(커밋 해시)을 적으세요. 비우면 나중에 받은 모델이 달라져 점수를 다시 만들 수 없습니다.
- 장치는 `device`(기본 `auto`: GPU가 있으면 GPU)입니다. `dtype: auto`는 GPU가 새 것(Ampere 이상)이면 bfloat16, 오래된 GPU(예: Colab 무료의 T4)면 float16, CPU면 float32를 고릅니다. float16은 일부 모델에서 수치가 불안정할 수 있으니 기본 설정처럼 `float32`를 직접 적는 편이 안전합니다.
- ngram 점수와 neural 점수는 비교할 수 없습니다. 그래서 결과 폴더를 나눴고, 섞어서 `compare`하면 리포트에 경고가 뜹니다. 모델을 바꾼 neural끼리도 마찬가지입니다.
- 어떤 모델로 채점했는지는 `results*/<이름>/run_meta.json`과 리포트에 남습니다.
- 평가 대상 모델과 같은 계열을 FM으로 쓰면 그 모델의 문체에 점수가 후할 수 있습니다.

### 필요한 메모리

모델 파일 크기는 허깅페이스에서 조회한 값이고, 실제 사용량은 여기에 약간의 작업 공간이 더해집니다.

| 구성 | GPU 메모리(대략) |
|---|---|
| Qwen2.5-1.5B + bge-m3, float32 (기본) | 약 8.5GB |
| 같은 구성, float16 | 약 4.2GB |
| Qwen2.5-7B + bge-m3, 16bit | 약 16GB 이상 (T4에는 안 들어갑니다) |

### Google Colab 무료 플랜에서 돌리기

런타임 유형을 GPU(T4)로 바꾼 뒤 아래 순서로 실행합니다. 무료 플랜은 GPU 배정이 보장되지 않고, 세션이 끝나면 디스크가 사라집니다. 이 절차는 이 저장소에서 GPU로 직접 돌려 확인한 것이 아닙니다.

```python
# 1) 저장소와 설치. torch, transformers는 Colab에 이미 있으니 requirements-neural.txt는 설치하지 않습니다.
!git clone <이 저장소 주소> FineTuningEval
%cd FineTuningEval
!pip install -r requirements.txt

# 2) 드라이브를 연결하고, 평가 데이터와 모델 출력을 가져옵니다(저장소에는 없습니다).
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p dataset/labeled_questionnaire_100_handoff/data runs/base
!cp "/content/drive/MyDrive/<폴더>/extraction.jsonl" "/content/drive/MyDrive/<폴더>/response.jsonl" dataset/labeled_questionnaire_100_handoff/data/
!cp "/content/drive/MyDrive/<폴더>/base/"*.jsonl runs/base/

# 3) 모델을 세션마다 다시 내려받지 않도록 캐시를 드라이브에 둡니다.
%env HF_HOME=/content/drive/MyDrive/hf_cache

# 4) 채점
!python -m scorer --config config.neural.json validate
!python -m scorer --config config.neural.json score base --model-id <모델 이름>

# 5) 결과는 세션이 끝나면 사라지니 드라이브로 옮깁니다.
!cp -r results_neural /content/drive/MyDrive/
```

`ft_seed42` 같은 실행도 `runs/ft_seed42/`에 같은 방식으로 두고 `score ft_seed42`를 하면 됩니다.
