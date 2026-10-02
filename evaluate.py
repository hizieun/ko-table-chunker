"""파서 검증 하네스 — 3층.

  L1 불변식      : 정답 라벨 0개. 모든 문서에 공짜. 파서 버그의 대부분을 여기서 잡음.
  L2 검색 평가   : 표에서 질의를 자동 생성 -> Recall@k / MRR / 정답값 포함률
  L3 음성 통제   : 표 셀을 섞어도 점수가 안 떨어지면 지표가 구조를 안 보고 있다는 뜻

핵심 규율: 임베더는 고정하고 파서/청커만 바꾼다. 안 그러면 델타 귀인이 불가능.

실행:  python3 evaluate.py              (오프라인, char-ngram TF-IDF)
       python3 evaluate.py --st BAAI/bge-m3   (sentence-transformers 로 교체)
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from dataclasses import dataclass

import numpy as np
from bs4 import BeautifulSoup

from html_chunker import (BACKEND, FIXTURE, Chunk, build_table, clean, invariants,
                          header_cols, is_prose_row, is_value, parse, row_kv,
                          to_markdown)

# --------------------------------------------------------------------------- #
# 임베더 (교체 가능. 평가 중에는 절대 바꾸지 말 것)
# --------------------------------------------------------------------------- #

class Retriever:
    """검색기 공통 인터페이스. 점수가 아니라 **순위**를 낸다.

    RRF 융합과 리랭커는 점수 스케일이 서로 달라 더할 수 없다. 순위로 통일하면
    어떤 조합이든 같은 방식으로 섞인다.
    """

    def ranks(self, queries: list[str], docs: list[str]) -> np.ndarray:
        self.fit(docs)
        return np.argsort(-(self.encode(queries) @ self.encode(docs).T), axis=1)


class CharTfidf(Retriever):
    """한국어 char 3-gram TF-IDF. 다운로드 0, 즉시 실행. 어휘적 베이스라인."""

    def __init__(self, n: int = 3):
        self.n, self.vocab, self.idf = n, {}, None

    def _grams(self, s: str) -> list[str]:
        s = re.sub(r"\s+", " ", s.lower())
        return [s[i:i + self.n] for i in range(max(0, len(s) - self.n + 1))]

    def fit(self, docs: list[str]):
        self.vocab = {g: i for i, g in enumerate({g for d in docs for g in self._grams(d)})}
        m = self._counts(docs)
        df = (m > 0).sum(0)
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1.0
        return self

    def _counts(self, docs: list[str]) -> np.ndarray:
        m = np.zeros((len(docs), len(self.vocab)), dtype=np.float32)
        for r, d in enumerate(docs):
            for g in self._grams(d):
                j = self.vocab.get(g)
                if j is not None:
                    m[r, j] += 1
        return m

    def encode(self, docs: list[str]) -> np.ndarray:
        v = self._counts(docs) * self.idf
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.clip(n, 1e-9, None)


class STEmbedder(Retriever):
    def __init__(self, name: str):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)

    def fit(self, docs):
        return self

    def encode(self, docs):
        return self.m.encode(docs, normalize_embeddings=True, show_progress_bar=False)


class HybridRRF(Retriever):
    """Reciprocal Rank Fusion. 어휘 + 밀집을 순위로 섞는다.

    한국어는 조사가 붙어 어휘 매칭이 깎이고, 반대로 숫자·고유명사는 밀집이 약하다.
    표 검색은 둘 다 필요해서 융합 이득이 크다. 다운로드·학습 0.

    측정(40문서/질의 397): 어휘 R@1 0.358, 밀집 0.353 -> 융합 0.438 (+22%).
    """

    def __init__(self, *parts: Retriever, k: int = 60):
        self.parts, self.k = parts, k

    def ranks(self, queries, docs):
        n = len(docs)
        score = np.zeros((len(queries), n))
        for p in self.parts:
            for qi, r in enumerate(p.ranks(queries, docs)):
                score[qi, r] += 1.0 / (self.k + np.arange(n) + 1)
        return np.argsort(-score, axis=1)


class Reranked(Retriever):
    """교차 인코더 리랭커. base 의 상위 top_n 만 다시 정렬한다.

    한국어 리랭커는 전부 1~2GB 다운로드가 필요해 기본값으로 두지 않았다.
    쓰려면 --rerank 로 모델명을 넘길 것 (예: dragonkue/bge-reranker-v2m3-ko).
    """

    def __init__(self, base: Retriever, model: str, top_n: int = 30):
        from sentence_transformers import CrossEncoder
        self.base, self.top_n = base, top_n
        self.ce = CrossEncoder(model)

    def ranks(self, queries, docs):
        base = self.base.ranks(queries, docs)
        out = []
        for q, r in zip(queries, base):
            head = list(r[:self.top_n])
            s = self.ce.predict([(q, docs[i]) for i in head])
            out.append(np.array([head[j] for j in np.argsort(-s)] + list(r[self.top_n:])))
        return np.array(out)


# --------------------------------------------------------------------------- #
# 비교 대상 청킹 전략
# --------------------------------------------------------------------------- #

def strat_flat(html: str, max_chars: int) -> list[Chunk]:
    """대부분의 팀이 실제로 배포하는 베이스라인: get_text() + 고정 길이 분할."""
    txt = clean(BeautifulSoup(html, BACKEND).get_text(" "))
    return [Chunk(text=txt[i:i + max_chars], context=txt[i:i + max_chars], kind="text")
            for i in range(0, len(txt), max_chars)]


def strat_markdown(html: str, max_chars: int) -> list[Chunk]:
    """표를 마크다운으로 보존하되 행 원자성은 없음 (길이로만 자름)."""
    soup = BeautifulSoup(html, BACKEND)
    parts = []
    for t in soup.find_all("table"):
        if t.find_parent("table") is None:
            tb = build_table(t)
            parts.append(to_markdown(tb.labels, tb.body))
            t.decompose()
    parts.insert(0, clean(soup.get_text(" ")))
    blob = "\n".join(p for p in parts if p)
    return [Chunk(text=blob[i:i + max_chars], context=blob[i:i + max_chars], kind="text")
            for i in range(0, len(blob), max_chars)]


def strat_stc(html: str, max_chars: int) -> list[Chunk]:
    return parse(html, max_chars=max_chars)


STRATEGIES = {"flat": strat_flat, "markdown": strat_markdown, "stc": strat_stc}


# --------------------------------------------------------------------------- #
# 질의 자동 생성 (정답 = 셀 좌표)
# --------------------------------------------------------------------------- #

@dataclass
class Query:
    q: str
    keys: list[str]      # 행을 식별하는 값들 (좌측 헤더축 전부). 포맷 독립 gold
    label: str           # 정답 셀의 컬럼 라벨
    answer: str          # 정답 셀 값
    scope: str           # 문서 식별자 (caption) — 문서 간 오답 gold 방지
    row_chars: int       # 정답 행 1줄의 길이. 신호밀도의 분자


def gen_queries(html: str, per_table: int = 6, seed: int = 0) -> list[Query]:
    """표의 각 행에서 '키 컬럼 -> 타깃 컬럼' 질의를 만든다.

    주의: 템플릿 질의는 셀 문자열을 그대로 쓰므로 '패러프레이즈 강건성'은 측정하지
    못한다. 그건 임베더의 몫이고, 여기서 재려는 건 '행이 청킹을 살아남았는가'다.
    패러프레이즈까지 보려면 같은 (answer, row_kv) 에 LLM 으로 질의문만 다시 써 붙일 것.
    """
    rng = random.Random(seed)
    soup = BeautifulSoup(html, BACKEND)
    # 문서를 특정하는 말이 질의에 없으면 답이 불가능하다. 실제 코퍼스는 비슷한 표를
    # 문서마다 반복하므로(같은 caption, 같은 기관명, 다른 숫자) caption 만으로는
    # 모자란다. 제목을 scope 로 쓴다 — 실제 사용자도 "가온증권의 ..." 라고 묻는다.
    head = soup.find(["h1", "h2"]) or soup.title
    doc_title = clean(head.get_text(" ", strip=True)) if head else ""
    out: list[Query] = []
    for tag in soup.find_all("table"):
        if tag.find_parent("table") is not None:
            continue
        t = build_table(tag)
        if not t.body:
            continue
        hc = header_cols(t)
        cands = []
        for row in t.body:
            if is_prose_row(row):
                continue        # 표 안의 안내문 행. 질문거리가 아니다
            # colspan 으로 복제된 칸은 제외 — row_kv 가 원본 컬럼에만 값을 내므로
            # 복제된 칸을 정답으로 잡으면 도달 불가능한 gold 가 된다 (없는 결함 보고)
            own = [i for i, c in enumerate(row) if c.text and c.origin[1] == i]
            # 행을 식별하는 건 좌측 헤더축 '전부' 다. 첫 칸만 쓰면 '전 용' 처럼
            # 이용료/이용범위 두 행에 모두 걸리는 중의적 질의가 만들어진다.
            key_is = [i for i in own if i < hc]
            tgt = [i for i in own if i >= hc and t.labels[i]]
            if not key_is or not tgt:
                continue
            i = rng.choice(tgt)
            keys = [row[k].text for k in key_is]
            scope = " ".join(x for x in (doc_title, t.caption) if x) or "문서"
            cands.append(Query(
                q=f"{scope}에서 {' '.join(keys)} 의 "
                  f"{t.labels[i].replace(' > ', ' ')}은 얼마인가?",
                keys=keys, label=t.labels[i],
                answer=row[i].text, scope=scope,
                row_chars=len(row_kv(t.labels, row)),
            ))
        rng.shuffle(cands)
        out.extend(cands[:per_table])
    return out


# --------------------------------------------------------------------------- #
# 채점
# --------------------------------------------------------------------------- #

_MD_ROW = re.compile(r"^\s*\|(.+)\|\s*$", re.M)
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def recoverable(text: str, q: Query) -> bool:
    """이 청크만 보고 (라벨 -> 정답값) 연결을 기계적으로 복원할 수 있는가?

    단순 문자열 포함은 쓸모가 없다. get_text() 로 눌러버린 청크에도 라벨 문자열과
    값 문자열이 '둘 다 어딘가' 존재하기 때문 — 하지만 어느 값이 어느 컬럼인지는 잃었다.
    이 함수는 LLM 추출 판정(L4)의 결정론적 대체물이다. 실제 LLM 판정으로 바꾸려면
    이 함수만 교체하면 된다.
    """
    if q.answer not in text or any(k not in text for k in q.keys):
        return False
    # (a) KV 선형화 형태 — key 와 (label: answer) 가 **같은 줄**이어야 한다.
    #     청크 전체에서 찾으면 표 하나가 통째로 든 청크는 무조건 통과해버린다.
    #     '전용 이용료'를 묻는데 '표준' 줄의 값을 답해도 잡히지 않는다.
    pat = re.compile(rf"{re.escape(q.label)}\s*:\s*{re.escape(q.answer)}")
    for line in text.splitlines():
        if pat.search(line) and all(k in line for k in q.keys):
            return True
    # (b) 마크다운 표 형태 — 헤더에서 컬럼 위치를 찾고 같은 위치의 본문 셀과 대조
    rows = [[c.strip() for c in m.group(1).split("|")] for m in _MD_ROW.finditer(text)]
    rows = [r for r in rows if not all(set(c) <= {"-", ":", ""} for c in r)]
    if len(rows) >= 2 and q.label in rows[0]:
        j = rows[0].index(q.label)
        return any(len(r) > j and r[j] == q.answer
                   and all(any(k in cell for cell in r) for k in q.keys)
                   for r in rows[1:])
    return False


def distractors(text: str, q: Query) -> int | None:
    """정답이 들어있는 '줄' 안에, 정답 말고 값처럼 생긴 것이 몇 개나 더 있는가.

    recoverable() 은 완벽한 추출기를 가정하므로 이 실패를 못 잡는다 — 라벨이
    붙어 있으면 기계적으로는 복원 가능하기 때문이다. 실제 LLM 이 틀리는 원인은
    정보 손실이 아니라 **한 줄 안의 혼동값 밀도**다. 행 KV 는 한 줄에 값이 4개,
    셀 단위는 1개다. 낮을수록 좋다. L4(실제 LLM 추출)의 결정론적 선행지표.
    """
    own = set(_NUM.findall(q.answer))
    best = None
    for line in text.splitlines():
        if q.answer not in line or any(k not in line for k in q.keys):
            continue
        # ponytail: 라벨에 든 연도('2024년')도 세어진다. 모든 전략에 똑같이 얹히는
        # 상수항이라 비교는 공정하다. 절대값이 필요해지면 라벨 구간을 빼고 셀 것.
        n = sum(1 for x in _NUM.findall(line) if x not in own)
        best = n if best is None else min(best, n)
    return best


def score(qs: list[Query], chunks: list[Chunk], emb,
          budgets=(1200, 3000), ks=(1, 5)) -> dict:
    """컨텍스트 예산으로 정규화해 채점한다.

    고정 k 의 Recall@k 는 청크가 클수록 유리해 전략 비교를 오염시킨다(granularity
    confound). 같은 글자 수를 LLM 에 넣었을 때 답할 수 있느냐로 맞춰야 공정하다.
    """
    if not chunks:
        return {"note": "no chunks"}
    docs = [c.text for c in chunks]
    rank = emb.ranks([q.q for q in qs], docs)

    hit_b = {b: 0 for b in budgets}
    hit_k = {k: 0 for k in ks}
    rr, broken, dens = 0.0, 0, 0.0
    dist, dist_n = 0, 0
    for r, q in zip(rank, qs):
        gold = {i for i, c in enumerate(chunks) if recoverable(c.text, q)}
        if not gold:
            broken += 1          # 어떤 청크로도 답이 복원 불가 = 청킹이 행을 파괴
            continue
        for k in ks:
            hit_k[k] += any(i in gold for i in r[:k])
        for b in budgets:
            used, ok = 0, False
            for i in r:
                if used + len(docs[i]) > b and used > 0:
                    break
                used += len(docs[i])
                if i in gold:
                    ok = True
            hit_b[b] += ok
            if b == budgets[0]:
                dens += (q.row_chars / max(1, used)) if ok else 0.0
        pos = next((j for j, i in enumerate(r) if i in gold), None)
        rr += 1.0 / (pos + 1) if pos is not None else 0.0
        d = next((x for x in (distractors(docs[i], q) for i in sorted(gold))
                  if x is not None), None)
        if d is not None:
            dist, dist_n = dist + d, dist_n + 1

    n = max(1, len(qs))
    return {
        "행파괴율": broken / n,
        **{f"Rec@{b}자": hit_b[b] / n for b in budgets},
        **{f"R@{k}": hit_k[k] / n for k in ks},
        "MRR": rr / n,
        "신호밀도": dens / n,
        "혼동값": dist / max(1, dist_n),
        "청크수": len(chunks),
        "평균길이": round(sum(len(d) for d in docs) / len(docs)),
    }


def shuffle_cells(html: str, seed: int = 7) -> str:
    """L3 음성 통제: 표 내부 셀 값만 섞는다. 텍스트 총량·토큰 분포는 동일."""
    rng = random.Random(seed)
    soup = BeautifulSoup(html, BACKEND)
    for t in soup.find_all("table"):
        cells = [c for c in t.find_all(["td", "th"]) if c.find_parent("table") is t]
        texts = [c.get_text(" ", strip=True) for c in cells]
        rng.shuffle(texts)
        for c, s in zip(cells, texts):
            c.string = s
    return str(soup)


# --------------------------------------------------------------------------- #

KEYS = ["행파괴율", "Rec@3000자", "R@1", "MRR", "혼동값", "신호밀도", "청크수", "평균길이"]


def _row(name: str, m: dict) -> str:
    cells = []
    for k in KEYS:
        v = m.get(k, "-")
        cells.append(f"{v:>11.3f}" if isinstance(v, float) else f"{v:>11}")
    return f"{name:<10}" + "".join(cells)


def build_retriever(a) -> Retriever:
    """CLI 플래그 -> 검색기 조합. 파이프라인에서도 그대로 쓰면 된다."""
    lex = CharTfidf()
    ret: Retriever = lex
    if a.st:
        ret = HybridRRF(lex, STEmbedder(a.st)) if a.hybrid else STEmbedder(a.st)
    elif a.hybrid:
        raise SystemExit("--hybrid 는 --st 와 함께 써야 한다 (섞을 밀집 검색기가 필요)")
    if a.rerank:
        ret = Reranked(ret, a.rerank, a.rerank_top)
    return ret


def describe(r: Retriever) -> str:
    if isinstance(r, Reranked):
        return f"{describe(r.base)} + 리랭커({r.ce.model_card_data.base_model or '?'})"
    if isinstance(r, HybridRRF):
        return "RRF(" + " + ".join(describe(p) for p in r.parts) + ")"
    return type(r).__name__


def run(docs: list[str], fn, max_chars: int) -> list[Chunk]:
    out = []
    for d in docs:
        out.extend(fn(d, max_chars))
    return out


def l1_only(paths: list[str], max_chars: int = 900) -> None:
    """L1 만 돌리고 **텍스트가 한 글자도 없는** 요약을 낸다.

    보안상 문서를 반출할 수 없는 환경에서, 사내에서 돌린 뒤 요약만 공유하기 위한
    모드다. L2/L3 는 질의를 생성하므로 출력에 실문서 내용이 섞인다 — 그래서 뺀다.
    네트워크·LLM·임베딩 모델·정답 라벨 모두 필요 없다.
    """
    import collections
    import pathlib
    from html_chunker import build_table, header_cols, is_crosstab, _int, _own

    n = len(paths)
    agg = collections.Counter()
    cov, worst, fails = [], ("", 1.0), collections.defaultdict(list)
    spans, hdr, shapes = collections.Counter(), collections.Counter(), []

    for p in paths:
        html = pathlib.Path(p).read_text(encoding="utf-8", errors="replace")
        inv = invariants(html, max_chars=max_chars)
        for k in ("rectangular", "cell_conservation", "row_atomic"):
            agg[k] += int(inv[k])
            if not inv[k]:
                fails[k].append(p)
        agg["n_chunks"] += inv["n_chunks"]
        agg["n_tables"] += inv["n_tables"]
        agg["oversize"] += inv["oversize_chunks"]
        cov.append(inv["text_coverage"])
        if inv["text_coverage"] < worst[1]:
            worst = (p, inv["text_coverage"])
        if inv["text_coverage"] <= 0.98:
            fails["text_coverage"].append(f"{p} ({inv['text_coverage']:.4f}, "
                                          f"유실 {len(inv['missing_tokens'])}+)")

        soup = BeautifulSoup(html, BACKEND)
        for tag in soup.find_all("table"):
            for c in tag.find_all(["td", "th"]):
                spans[f"rs{max(1,_int(c.get('rowspan'),1))}×cs"
                      f"{max(1,_int(c.get('colspan'),1))}"] += 1
            if tag.find_parent("table") is None:
                t = build_table(tag)
                if t.grid:
                    shapes.append((len(t.grid), len(t.grid[0])))
                    hdr[f"헤더행{t.n_header}/헤더열{header_cols(t)}"
                        f"/{'교차표' if is_crosstab(t) else '목록'}"] += 1
        agg["mso"] += int("MsoNormalTable" in html)
        agg["th"] += int(bool(soup.find("th")))
        agg["nested"] += sum(1 for t in soup.find_all("table") if t.find_parent("table"))

    ok = lambda c: "✓" if c == n else "✗ FAIL"
    print("=" * 62)
    print("  보내도 되는 요약 — 문서 내용이 한 글자도 들어있지 않습니다")
    print("=" * 62)
    print(f"  문서                {n}")
    print(f"  표(최상위/중첩)      {agg['n_tables']} / {agg['nested']}")
    print(f"  청크                {agg['n_chunks']}  (oversize {agg['oversize']})")
    print(f"  Word 내보내기        {agg['mso']}/{n}      <th> 있는 문서  {agg['th']}/{n}")
    print()
    for k in ("rectangular", "cell_conservation", "row_atomic"):
        print(f"  {k:<18} {agg[k]}/{n}  {ok(agg[k])}")
    avg = sum(cov) / max(1, n)
    print(f"  {'text_coverage':<18} 평균 {avg:.4f} / 최저 {worst[1]:.4f}  "
          f"{'✓' if avg > 0.98 and worst[1] > 0.98 else '✗ FAIL'}")
    print()
    print(f"  표 모양 상위       {collections.Counter(shapes).most_common(6)}")
    print(f"  헤더 판정          {dict(hdr.most_common(8))}")
    print(f"  병합 분포          {dict(spans.most_common(8))}")
    print("=" * 62)

    if any(fails.values()):
        print("\n--- 아래는 로컬 확인용입니다. 파일명이 들어있으니 공유 전 확인하세요 ---")
        for k, v in fails.items():
            for f in v[:10]:
                print(f"  ✗ {k}: {f}")
            if len(v) > 10:
                print(f"     … 외 {len(v)-10}건")
    else:
        print("\n모든 불변식 통과 — 이 문서들에서 파서가 데이터를 잃거나 깨뜨리지 않았습니다.")


def report(docs: list[str], emb, max_chars: int = 900):
    print(f"\n== L1 구조 불변식 (라벨 불필요, 문서 {len(docs)}개) ==")
    agg = {}
    for d in docs:
        for k, v in invariants(d, max_chars=max_chars).items():
            if k == "missing_tokens":
                continue
            agg[k] = agg.get(k, 0) + (v if not isinstance(v, bool) else int(v))
    for k, v in agg.items():
        if k in ("rectangular", "cell_conservation", "row_atomic"):
            print(f"  {k:<18} {v}/{len(docs)}" + ("  ✓" if v == len(docs) else "  ✗ FAIL"))
        elif k == "text_coverage":
            r = v / len(docs)
            print(f"  {k:<18} {r:.4f}" + ("  ✓" if r > 0.98 else "  ✗ FAIL"))
        else:
            print(f"  {k:<18} {v}")

    qs = [q for d in docs for q in gen_queries(d)]
    print(f"\n== L2 검색 평가 (자동생성 질의 {len(qs)}개) ==")
    print(f"{'전략':<10}" + "".join(f"{k:>11}" for k in KEYS))
    base = {}
    for name, fn in STRATEGIES.items():
        base[name] = score(qs, run(docs, fn, max_chars), emb)
        print(_row(name, base[name]))
        if base[name]["청크수"] <= 5:
            print(f"    ⚠ 청크가 {base[name]['청크수']}개뿐 — R@5 는 무의미. 코퍼스를 키울 것")

    print("\n== L3 음성 통제 (표 셀 셔플) ==")
    sh = [shuffle_cells(d) for d in docs]
    for name, fn in STRATEGIES.items():
        m = score(qs, run(sh, fn, max_chars), emb)
        b, drop = base[name]["MRR"], base[name]["MRR"] - m["MRR"]
        if b < 0.05:
            ok = f"– 바닥값이라 통제 불가 (행파괴율 {base[name]['행파괴율']:.2f})"
        elif drop > 0.05:
            ok = "✓ 구조에 반응"
        else:
            ok = "✗ 지표가 구조를 못 봄 — 이 전략의 점수는 신뢰 불가"
        print(f"  {name:<12} MRR {b:.3f} -> {m['MRR']:.3f}  (Δ{drop:+.3f})  {ok}")


SEGMENTS = ["메모리", "파운드리", "시스템LSI", "OLED", "LCD", "생활가전", "영상디스플레이", "네트워크"]
CORPS = ["가온전자", "한빛소재", "대명화학", "서진테크", "누리정밀", "명진산업",
         "태성에너지", "율촌바이오", "동방물산", "선진기전", "다산반도체", "해성정공"]


def corpus(n_docs: int = 12, groups: int = 20, seed: int = 0) -> list[str]:
    """평가용 합성 코퍼스. 문서마다 회사·연도·숫자가 달라 distractor 가 실제로 존재한다.

    ponytail: 실데이터가 생기면 이걸 버리고 실문서를 넣을 것. 이건 하네스가
    돌아간다는 것과 전략 간 상대 순위를 보기 위한 최소 장치.
    """
    docs = []
    for d in range(n_docs):
        corp, year = CORPS[d % len(CORPS)], 2019 + d % 6
        rows = []
        for g in range(groups):
            for j in range(3):
                i = g * 3 + j
                first = f'<td rowspan="3">{SEGMENTS[g % len(SEGMENTS)]}</td>' if j == 0 else ""
                rows.append(
                    f"<tr>{first}<td>{1000 + d * 911 + i * 37:,}</td>"
                    f"<td>{900 + d * 733 + i * 23:,}</td>"
                    f"<td>{(i * 7 + d) % 40 - 10}.{(i + d) % 10}%</td></tr>")
        docs.append(f"""
<h1>{corp} {year} 사업보고서</h1>
<h2>3. 재무 현황</h2>
<p>{corp}의 {year}년 실적은 전년 대비 개선되었습니다. 주력 사업의 수요 회복과
원가 절감이 동시에 작용했으며, 부문별 세부 내역은 아래 표와 같습니다.
일부 부문은 환율 영향으로 변동성이 확대되었습니다.</p>
<table>
  <caption>{corp} {year} 부문별 매출 (단위: 억원)</caption>
  <tr><th rowspan="2">사업부문</th><th colspan="2">{year}년</th><th rowspan="2">증감률</th></tr>
  <tr><th>상반기</th><th>하반기</th></tr>
  {''.join(rows)}
</table>
<p>{corp}는 차년도에도 해당 기조를 유지할 계획입니다.</p>""")
    return docs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="HTML 파일들. 없으면 합성 코퍼스")
    ap.add_argument("--st", help="sentence-transformers 모델명 (예: BAAI/bge-m3)")
    ap.add_argument("--hybrid", action="store_true",
                    help="어휘 + 밀집 RRF 융합 (--st 와 함께). 다운로드 없이 R@1 +22%%")
    ap.add_argument("--rerank", metavar="MODEL",
                    help="교차 인코더 리랭커. 1~2GB 다운로드가 발생한다 "
                         "(예: dragonkue/bge-reranker-v2m3-ko)")
    ap.add_argument("--rerank-top", type=int, default=30)
    ap.add_argument("--max-chars", type=int, default=900)
    ap.add_argument("--l1", action="store_true",
                    help="L1 만. 텍스트 없는 요약만 출력 (네트워크·LLM 불필요)")
    a = ap.parse_args()

    if a.l1:
        if not a.paths:
            ap.error("--l1 은 파일 경로가 필요하다")
        l1_only(a.paths, a.max_chars)
        sys.exit()

    docs = [open(p, encoding="utf-8").read() for p in a.paths] if a.paths else corpus()
    ret = build_retriever(a)
    print(f"검색기: {describe(ret)}")
    report(docs, ret, a.max_chars)
