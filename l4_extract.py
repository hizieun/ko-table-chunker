"""L4 — 종단 추출 정확도. 실제 LLM 이 청크를 보고 값을 맞히는가.

L1~L3 은 전부 '정보가 보존됐는가' 를 본다. recoverable() 조차 **완벽한 추출기를
가정**하므로, 라벨만 붙어 있으면 통과시킨다. 하지만 실제로 틀리는 원인은 정보
손실이 아니라 한 줄 안의 혼동값 밀도다. 그건 LLM 을 실제로 돌려야만 확인된다.

표 질의는 LLM-as-judge 보다 **값 정확 일치**가 더 정확하고 싸다. 대신 한국어
숫자 표기를 정규화해야 한다 — '연간 55,000원' 과 '55000' 은 같은 값이다.

실행:
    python3 l4_extract.py fixtures/*.html --backend anthropic
    python3 l4_extract.py fixtures/05_워드내보내기.html --backend cmd \
        --cmd 'claude -p --model claude-haiku-4-5-20251001'
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import unicodedata

from evaluate import (STRATEGIES, Retriever, build_retriever, describe,
                      gen_queries)
from html_chunker import Chunk

# --------------------------------------------------------------------------- #
# 한국어 수량 정규화
# --------------------------------------------------------------------------- #

_BIG = {"조": 10**12, "억": 10**8, "만": 10**4}      # 그룹 구분자
_SMALL = {"천": 1000, "백": 100, "십": 10}           # 그룹 내 자릿수
_NEG = "-−△▲"
_TOK = re.compile(r"\d+(?:\.\d+)?|[조억만천백십]")


def normalize_num(s: str) -> float | None:
    """'연간 55,000원' / '55000' / '3.5억' / '1억 2천만' -> 같은 축의 float.

    한글 수사는 자리올림이 중첩된다: '1억 2천만' = 1e8 + (2*1000)*1e4 = 1.2e8.
    단순히 (숫자, 단위) 쌍을 더하면 '2천만' 이 2.0 이 되는 천만 배 오차가 나고,
    그게 조용히 '값' 으로 통과한다 — L4 채점이 통째로 거짓이 된다.
    그래서 조/억/만을 그룹 구분자로 두고 그룹 안을 따로 누적한다.
    """
    if s is None:
        return None
    s = unicodedata.normalize("NFKC", str(s)).strip()
    neg = bool(s) and s[0] in _NEG
    s = s.lstrip(_NEG).replace(",", "")

    toks = _TOK.findall(s)
    if not toks:
        return None
    if not any(t[0].isdigit() for t in toks):
        # 숫자가 하나도 없으면 문자열 전체가 수사일 때만 수로 본다.
        # '백만' 은 1,000,000 이지만 '만족' 은 10,000 이 아니다.
        if re.sub(r"[조억만천백십\s]", "", s):
            return None
    total = group = 0.0
    cur: float | None = None
    for t in toks:
        if t in _BIG:
            group += cur if cur is not None else (0.0 if group else 1.0)
            total += group * _BIG[t]
            group, cur = 0.0, None
        elif t in _SMALL:
            group += (cur if cur is not None else 1.0) * _SMALL[t]
            cur = None
        else:
            cur = float(t)
    total += group + (cur or 0.0)
    return -total if neg else total


def same_value(a: str, b: str) -> bool:
    """정답 판정. 숫자면 정규화 비교, 아니면 공백 제거 후 문자열 비교."""
    na, nb = normalize_num(a), normalize_num(b)
    if na is not None and nb is not None:
        return abs(na - nb) < 1e-9
    norm = lambda x: re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(x or "")))
    return norm(a) == norm(b) and bool(norm(a))


# --------------------------------------------------------------------------- #
# LLM 백엔드 (교체 가능)
# --------------------------------------------------------------------------- #

PROMPT = """아래 문서 조각만 보고 질문에 답하세요.

[문서]
{ctx}

[질문] {q}

값만 출력하세요. 설명·단위 설명·문장 금지. 찾을 수 없으면 정확히 NONE 만 출력."""


def _anthropic(prompt: str, model: str) -> str:
    from anthropic import Anthropic
    r = Anthropic().messages.create(
        model=model, max_tokens=64,
        messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in r.content if b.type == "text").strip()


def _bedrock(prompt: str, model: str) -> str:
    import boto3
    r = boto3.client("bedrock-runtime").invoke_model(
        modelId=model,
        body=json.dumps({"anthropic_version": "bedrock-2023-05-31",
                         "max_tokens": 64,
                         "messages": [{"role": "user", "content": prompt}]}))
    return json.loads(r["body"].read())["content"][0]["text"].strip()


def _cmd(prompt: str, model: str, cmd: str = "", tries: int = 3) -> str:
    """임의 CLI 를 판정기로. 프롬프트는 stdin 으로 넣는다.

    CLI 판정기는 프로세스를 새로 띄우므로 가끔 느리다. 재시도 없이 쓰면
    타임아웃이 **오답으로 집계되어** 전략 간 차이로 둔갑한다 — 실제로
    언피벗 A/B 에서 차이의 절반이 타임아웃이었다.
    """
    last = None
    for i in range(tries):
        try:
            r = subprocess.run(cmd, shell=True, input=prompt, capture_output=True,
                               text=True, timeout=120 * (i + 1))
            if r.stdout.strip():
                return r.stdout.strip()
            last = RuntimeError(f"빈 응답: {r.stderr.strip()[:80]}")
        except subprocess.TimeoutExpired as e:
            last = e
    raise last


BACKENDS = {"anthropic": _anthropic, "bedrock": _bedrock, "cmd": _cmd}
DEFAULT_MODEL = {"anthropic": "claude-haiku-4-5-20251001",
                 "bedrock": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                 "cmd": ""}


# --------------------------------------------------------------------------- #

def retrieve(chunks: list[Chunk], ranking, budget: int = 3000) -> str:
    """예산만큼 상위 청크를 모아 컨텍스트로. L2 와 같은 규칙(예산 정규화).

    LLM 에는 임베딩용 text 가 아니라 context(표는 마크다운)를 넣는다 — 이중 표현.
    """
    out, used = [], 0
    for i in ranking:
        t = chunks[i].context or chunks[i].text
        if used + len(t) > budget and out:
            break
        out.append(t)
        used += len(t)
    return "\n\n".join(out)


def run(docs: list[str], backend: str, model: str, cmd: str = "",
        max_chars: int = 900, limit: int | None = None,
        retriever: Retriever | None = None, verbose: bool = True) -> dict:
    from evaluate import CharTfidf
    qs = [q for d in docs for q in gen_queries(d)]
    if limit:
        qs = qs[:limit]
    ret = retriever or CharTfidf()
    call = BACKENDS[backend]
    results = {}
    for name, fn in STRATEGIES.items():
        chunks = [c for d in docs for c in fn(d, max_chars)]
        rankings = ret.ranks([q.q for q in qs], [c.text for c in chunks])
        ok = 0
        for q, ranking in zip(qs, rankings):
            ctx = retrieve(chunks, ranking)
            kw = {"cmd": cmd} if backend == "cmd" else {}
            try:
                pred = call(PROMPT.format(ctx=ctx, q=q.q), model, **kw)
            except Exception as e:                    # 백엔드 실패는 오답이 아니다
                print(f"  ! {name}: {type(e).__name__}: {e}")
                break
            hit = same_value(pred, q.answer)
            ok += hit
            if not hit and verbose:
                print(f"  ✗ [{name}] {q.q[:60]}\n      정답={q.answer!r} 응답={pred[:60]!r}")
        results[name] = ok / max(1, len(qs))
        if verbose:
            print(f"{name:<10} 정답률 {results[name]:.3f}  ({ok}/{len(qs)})")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--backend", choices=list(BACKENDS), default="anthropic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--cmd", default="", help="--backend cmd 일 때 실행할 명령")
    ap.add_argument("--limit", type=int, help="질의 수 상한 (비용 통제)")
    ap.add_argument("--st", help="밀집 임베더 (예: BAAI/bge-m3)")
    ap.add_argument("--hybrid", action="store_true", help="어휘+밀집 RRF 융합")
    ap.add_argument("--rerank", metavar="MODEL", help="교차 인코더 리랭커 (1~2GB)")
    ap.add_argument("--rerank-top", type=int, default=30)
    a = ap.parse_args()

    docs = [open(p, encoding="utf-8").read() for p in a.paths]
    model = a.model or DEFAULT_MODEL[a.backend]
    ret = build_retriever(a)
    print(f"L4 종단 추출 — backend={a.backend} model={model or a.cmd}")
    print(f"검색기: {describe(ret)}")
    run(docs, a.backend, model, a.cmd, limit=a.limit, retriever=ret)
