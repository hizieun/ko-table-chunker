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

from html_chunker import (FIXTURE, Chunk, build_table, clean, invariants, parse,
                          row_kv, to_markdown)

# --------------------------------------------------------------------------- #
# 임베더 (교체 가능. 평가 중에는 절대 바꾸지 말 것)
# --------------------------------------------------------------------------- #

class CharTfidf:
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


class STEmbedder:
    def __init__(self, name: str):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)

    def fit(self, docs):
        return self

    def encode(self, docs):
        return self.m.encode(docs, normalize_embeddings=True, show_progress_bar=False)


# --------------------------------------------------------------------------- #
# 비교 대상 청킹 전략
# --------------------------------------------------------------------------- #

def strat_flat(html: str, max_chars: int) -> list[Chunk]:
    """대부분의 팀이 실제로 배포하는 베이스라인: get_text() + 고정 길이 분할."""
    txt = clean(BeautifulSoup(html, "html.parser").get_text(" "))
    return [Chunk(text=txt[i:i + max_chars], context=txt[i:i + max_chars], kind="text")
            for i in range(0, len(txt), max_chars)]


def strat_markdown(html: str, max_chars: int) -> list[Chunk]:
    """표를 마크다운으로 보존하되 행 원자성은 없음 (길이로만 자름)."""
    soup = BeautifulSoup(html, "html.parser")
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
    key: str             # 행을 식별하는 값 (예: "메모리")
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
    soup = BeautifulSoup(html, "html.parser")
    out: list[Query] = []
    for tag in soup.find_all("table"):
        if tag.find_parent("table") is not None:
            continue
        t = build_table(tag)
        if not t.body:
            continue
        cands = []
        for row in t.body:
            key_i = next((i for i, c in enumerate(row) if c.text), None)
            if key_i is None:
                continue
            tgt = [i for i, c in enumerate(row) if c.text and i != key_i and t.labels[i]]
            if not tgt:
                continue
            i = rng.choice(tgt)
            scope = t.caption or "문서"
            cands.append(Query(
                q=f"{scope}에서 {t.labels[key_i]} {row[key_i].text}의 "
                  f"{t.labels[i].replace(' > ', ' ')}은 얼마인가?",
                key=row[key_i].text, label=t.labels[i],
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


def recoverable(text: str, q: Query) -> bool:
    """이 청크만 보고 (라벨 -> 정답값) 연결을 기계적으로 복원할 수 있는가?

    단순 문자열 포함은 쓸모가 없다. get_text() 로 눌러버린 청크에도 라벨 문자열과
    값 문자열이 '둘 다 어딘가' 존재하기 때문 — 하지만 어느 값이 어느 컬럼인지는 잃었다.
    이 함수는 LLM 추출 판정(L4)의 결정론적 대체물이다. 실제 LLM 판정으로 바꾸려면
    이 함수만 교체하면 된다.
    """
    if q.key not in text or q.answer not in text:
        return False
    # (a) KV 선형화 형태
    if re.search(rf"{re.escape(q.label)}\s*:\s*{re.escape(q.answer)}", text):
        return True
    # (b) 마크다운 표 형태 — 헤더에서 컬럼 위치를 찾고 같은 위치의 본문 셀과 대조
    rows = [[c.strip() for c in m.group(1).split("|")] for m in _MD_ROW.finditer(text)]
    rows = [r for r in rows if not all(set(c) <= {"-", ":", ""} for c in r)]
    if len(rows) >= 2 and q.label in rows[0]:
        j = rows[0].index(q.label)
        return any(len(r) > j and r[j] == q.answer and q.key in r for r in rows[1:])
    return False


def score(qs: list[Query], chunks: list[Chunk], emb,
          budgets=(1200, 3000), ks=(1, 5)) -> dict:
    """컨텍스트 예산으로 정규화해 채점한다.

    고정 k 의 Recall@k 는 청크가 클수록 유리해 전략 비교를 오염시킨다(granularity
    confound). 같은 글자 수를 LLM 에 넣었을 때 답할 수 있느냐로 맞춰야 공정하다.
    """
    if not chunks:
        return {"note": "no chunks"}
    docs = [c.text for c in chunks]
    emb.fit(docs)
    D, Q = emb.encode(docs), emb.encode([q.q for q in qs])
    rank = np.argsort(-(Q @ D.T), axis=1)

    hit_b = {b: 0 for b in budgets}
    hit_k = {k: 0 for k in ks}
    rr, broken, dens = 0.0, 0, 0.0
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

    n = max(1, len(qs))
    return {
        "행파괴율": broken / n,
        **{f"Rec@{b}자": hit_b[b] / n for b in budgets},
        **{f"R@{k}": hit_k[k] / n for k in ks},
        "MRR": rr / n,
        "신호밀도": dens / n,
        "청크수": len(chunks),
        "평균길이": round(sum(len(d) for d in docs) / len(docs)),
    }


def shuffle_cells(html: str, seed: int = 7) -> str:
    """L3 음성 통제: 표 내부 셀 값만 섞는다. 텍스트 총량·토큰 분포는 동일."""
    rng = random.Random(seed)
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all("table"):
        cells = [c for c in t.find_all(["td", "th"]) if c.find_parent("table") is t]
        texts = [c.get_text(" ", strip=True) for c in cells]
        rng.shuffle(texts)
        for c, s in zip(cells, texts):
            c.string = s
    return str(soup)


# --------------------------------------------------------------------------- #

KEYS = ["행파괴율", "Rec@1200자", "Rec@3000자", "R@1", "R@5", "MRR", "신호밀도", "청크수", "평균길이"]


def _row(name: str, m: dict) -> str:
    cells = []
    for k in KEYS:
        v = m.get(k, "-")
        cells.append(f"{v:>11.3f}" if isinstance(v, float) else f"{v:>11}")
    return f"{name:<10}" + "".join(cells)


def run(docs: list[str], fn, max_chars: int) -> list[Chunk]:
    out = []
    for d in docs:
        out.extend(fn(d, max_chars))
    return out


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
    ap.add_argument("--max-chars", type=int, default=900)
    a = ap.parse_args()

    docs = [open(p, encoding="utf-8").read() for p in a.paths] if a.paths else corpus()
    report(docs, STEmbedder(a.st) if a.st else CharTfidf(), a.max_chars)
