"""역할 B의 문장 품질을 사전학습 모델로 재는 백엔드: 언어모델(FM)과 문장 임베딩 모델(AM).

lm.py의 BigramLM·LSI와 같은 계산(FM = exp(-|두 문장의 길이 정규화 로그확률 차이|), AM = 코사인 유사도에서
음수를 0으로 자른 값)을 하되, 확률과 벡터를 코퍼스가 아니라 사전학습 모델에서 얻는다. 코퍼스가 필요 없다.

torch와 transformers는 실제 모델을 불러올 때만 가져온다(load_*). 그래서 기본 설정(ngram)은 numpy만으로 돌아간다.
NeuralLM과 NeuralEmbedding은 token_logprobs(text)나 encode(text)만 있는 객체를 받으므로, 모델 없이도 시험할 수 있다.
"""
import math

import numpy as np

from .dataio import InputError

TINY = 1e-12


class NeuralLM:
    """토큰 로그확률을 내는 모델(source)로 FM을 계산한다.

    source.token_logprobs(text)는 문장 시작 표시 뒤 각 토큰의 로그확률 목록을 돌려준다(토큰이 없으면 빈 목록).
    """

    def __init__(self, source):
        """source를 저장하고 문장별 점수를 저장해 둘 칸을 만든다."""
        self.source = source
        self._cache = {}

    def log_prob(self, text):
        """토큰당 평균 로그확률 (1/N) * sum(log p). 토큰이 없으면 None."""
        if text not in self._cache:
            logprobs = list(self.source.token_logprobs(text))
            self._cache[text] = sum(logprobs) / len(logprobs) if logprobs else None
        return self._cache[text]

    def fm(self, candidate, reference):
        """길이로 정규화한 두 확률의 min/max. 로그로 계산한다. BigramLM.fm과 같은 식이다."""
        lc, lr = self.log_prob(candidate), self.log_prob(reference)
        if lc is None or lr is None:
            return 0.0
        return math.exp(-abs(lc - lr))


class NeuralEmbedding:
    """문장 벡터를 내는 모델(source)로 AM을 계산한다. source.encode(text)는 1차원 배열을 돌려준다."""

    def __init__(self, source):
        """source를 저장하고 문장별 벡터를 저장해 둘 칸을 만든다."""
        self.source = source
        self._cache = {}

    def vector(self, text):
        """문장의 벡터(float64 1차원 배열)."""
        if text not in self._cache:
            self._cache[text] = np.asarray(self.source.encode(text), dtype=np.float64)
        return self._cache[text]

    def am(self, candidate, reference):
        """AM. 두 문장 벡터의 코사인 유사도. 음수는 0으로 자르고, 벡터 크기가 1e-12 이하이면 0. LSI.am과 같다."""
        a, b = self.vector(candidate), self.vector(reference)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na <= TINY or nb <= TINY:
            return 0.0
        return max(0.0, float(a @ b) / (na * nb))


class NeuralSentenceScorer:
    """SentenceScorer와 같은 score()를 가진 neural 백엔드. 코퍼스를 쓰지 않으므로 oov 값은 None이다."""
    kind = "neural"

    def __init__(self, lm, embedding, info):
        """lm은 NeuralLM, embedding은 NeuralEmbedding, info는 실행 기록에 남길 모델 정보(dict)이다."""
        self.lm = lm
        self.embedding = embedding
        self.info = info

    def score(self, candidate, references):
        """후보의 FM, AM, oov_word(None), oov_bigram(None). 참조가 여럿이면 가장 높은 값(최댓값)을 쓴다."""
        if not isinstance(candidate, str) or not candidate.split():
            return {"FM": 0.0, "AM": 0.0, "oov_word": None, "oov_bigram": None}
        fm = max(self.lm.fm(candidate, r) for r in references)
        am = max(self.embedding.am(candidate, r) for r in references)
        return {"FM": fm, "AM": am, "oov_word": None, "oov_bigram": None}


def _require(spec, key, kind):
    """spec[key]가 문자열이어야 한다. 아니면 설정 오류."""
    value = spec.get(key)
    if not isinstance(value, str) or not value:
        raise InputError("config sentence_scorer.%s must be a non-empty string" % ".".join(kind + [key]))
    return value


def parse_spec(spec):
    """config의 sentence_scorer(dict)를 검사해 정리한 dict를 돌려준다. 모델 이름이 없으면 InputError."""
    if not isinstance(spec, dict) or not isinstance(spec.get("lm"), dict) or not isinstance(spec.get("embedder"), dict):
        raise InputError("config sentence_scorer of type neural needs 'lm' and 'embedder' objects")
    lm, emb = spec["lm"], spec["embedder"]
    pooling = emb.get("pooling", "cls")
    if pooling not in ("cls", "mean"):
        raise InputError("config sentence_scorer.embedder.pooling must be 'cls' or 'mean'")
    return {
        "type": "neural",
        "device": spec.get("device", "auto"),
        "lm": {"model": _require(lm, "model", ["lm"]), "revision": lm.get("revision"),
               "dtype": lm.get("dtype", "auto")},
        "embedder": {"model": _require(emb, "model", ["embedder"]), "revision": emb.get("revision"),
                     "dtype": emb.get("dtype", "auto"), "pooling": pooling,
                     "max_length": int(emb.get("max_length", 512))},
    }


def build_scorer(spec):
    """정리된 spec으로 실제 모델을 불러와 NeuralSentenceScorer를 만든다. torch와 transformers가 필요하다."""
    info = parse_spec(spec)
    device = _device(info["device"])
    lm = TorchCausalLM(info["lm"], device)
    embedder = TorchEmbedder(info["embedder"], device)
    info = dict(info, device=device)
    return NeuralSentenceScorer(NeuralLM(lm), NeuralEmbedding(embedder), info)


def _import_torch():
    """torch와 transformers를 가져온다. 없으면 설치하라는 안내와 함께 InputError."""
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise InputError("the neural sentence scorer needs torch and transformers (pip install torch transformers)") from exc
    return torch, transformers


def _device(name):
    """'auto'면 GPU가 있을 때 cuda, 없으면 cpu. 그 밖에는 받은 이름 그대로."""
    if name != "auto":
        return name
    torch, _ = _import_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype(torch, name, device):
    """dtype 이름을 torch dtype으로 바꾼다.

    'auto'는 cpu면 float32, cuda면 bf16을 기본으로 지원하는 GPU(Ampere 이상, 연산 능력 8 이상)일 때만
    bfloat16이고 그보다 오래된 GPU(예: Colab 무료의 T4)는 float16이다. 오래된 GPU에서 bfloat16은 느리거나
    지원되지 않는다. 수치가 민감하면 설정에서 float32를 직접 적는다.
    """
    if name == "auto":
        if not device.startswith("cuda"):
            return torch.float32
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    if not hasattr(torch, name) or not isinstance(getattr(torch, name), torch.dtype):
        raise InputError("unknown dtype %r in config sentence_scorer" % name)
    return getattr(torch, name)


def _dtype_kwargs(transformers, dtype):
    """from_pretrained에 dtype을 넘길 때 쓰는 인자 이름. transformers 4.56부터 dtype, 그 전에는 torch_dtype이다.

    버전을 읽을 수 없으면 새 이름(dtype)을 쓴다. 옛 이름은 4.56 이후 경고를 내고 나중에 없어질 수 있다.
    """
    parts = str(getattr(transformers, "__version__", "")).split(".")
    try:
        version = (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return {"dtype": dtype}
    return {"dtype": dtype} if version >= (4, 56) else {"torch_dtype": dtype}


class TorchCausalLM:
    """transformers의 인과 언어모델로 token_logprobs를 구한다. 모델은 평가 모드이고 기울기는 계산하지 않는다."""

    def __init__(self, spec, device):
        """spec['model']을 내려받아(또는 로컬 경로에서) 장치에 올린다. 문장 시작 표시는 bos, 없으면 eos를 쓴다."""
        torch, transformers = _import_torch()
        self.torch = torch
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            spec["model"], revision=spec["revision"], **_dtype_kwargs(transformers, _dtype(torch, spec["dtype"], device)))
        self.model.to(device).eval()
        self.device = device
        self.start_id = self.tokenizer.bos_token_id
        if self.start_id is None:
            self.start_id = self.tokenizer.eos_token_id
        if self.start_id is None:
            raise InputError("the language model %s has neither a bos nor an eos token" % spec["model"])

    def token_logprobs(self, text):
        """문장 시작 표시 뒤 각 토큰의 로그확률 목록. 토큰이 없으면 빈 목록."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            return []
        torch = self.torch
        inputs = torch.tensor([[self.start_id] + ids], device=self.device)
        with torch.no_grad():
            logits = self.model(inputs).logits[0, :-1].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        target = torch.tensor(ids, device=self.device)
        return logprobs.gather(1, target[:, None])[:, 0].tolist()


class TorchEmbedder:
    """transformers의 인코더로 문장 벡터를 구한다. 풀링은 'cls'(첫 토큰) 또는 'mean'(마스크 평균)이다."""

    def __init__(self, spec, device):
        """spec['model']을 내려받아(또는 로컬 경로에서) 장치에 올린다."""
        torch, transformers = _import_torch()
        self.torch = torch
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
        self.model = transformers.AutoModel.from_pretrained(
            spec["model"], revision=spec["revision"], **_dtype_kwargs(transformers, _dtype(torch, spec["dtype"], device)))
        self.model.to(device).eval()
        self.device = device
        self.pooling = spec["pooling"]
        self.max_length = spec["max_length"]

    def encode(self, text):
        """문장을 벡터(1차원 numpy 배열)로 바꾼다. 풀링 결과를 길이 1로 정규화한다."""
        torch = self.torch
        batch = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state
        if self.pooling == "cls":
            vector = hidden[:, 0]
        else:
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            vector = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        vector = torch.nn.functional.normalize(vector.float(), dim=-1)
        return vector[0].cpu().numpy()
