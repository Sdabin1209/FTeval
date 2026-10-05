"""모델 출력의 엄격한 JSON 읽기.

실패로 보는 경우: 어느 깊이에서든 중복 키, NaN/Infinity 리터럴, 무한대로 넘치는 숫자,
맨 앞 BOM, 값 뒤에 붙은 글자, 최상위 값이 객체가 아닌 경우.
"""
import json
import math


class StrictJSONError(ValueError):
    """엄격한 읽기에 실패했음을 나타내는 오류."""
    pass


def _no_duplicates(pairs):
    """같은 키가 두 번 나오면 실패로 만드는 json 읽기 훅."""
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise StrictJSONError("duplicate key: %r" % key)
        obj[key] = value
    return obj


def _bad_constant(name):
    """NaN, Infinity, -Infinity 리터럴을 실패로 만드는 json 읽기 훅."""
    raise StrictJSONError("literal not allowed: %s" % name)


def _parse_float(text):
    """실수를 읽되, 무한대로 넘치는 값(1e999)이면 실패로 만드는 json 읽기 훅."""
    value = float(text)
    if not math.isfinite(value):
        raise StrictJSONError("number overflows to infinity: %s" % text)
    return value


def parse_strict_object(text):
    """읽은 JSON 객체를 돌려주고, 실패하면 StrictJSONError를 낸다."""
    if not isinstance(text, str):
        raise StrictJSONError("not a string")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_no_duplicates,
            parse_constant=_bad_constant,
            parse_float=_parse_float,
        )
    except StrictJSONError:
        raise
    except (ValueError, RecursionError) as exc:  # JSONDecodeError는 ValueError의 하위 클래스
        raise StrictJSONError(str(exc)) from exc
    if not isinstance(value, dict):
        raise StrictJSONError("top-level value is not an object")
    return value
