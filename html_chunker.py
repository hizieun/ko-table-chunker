"""OCR HTML -> RAG 청크.

설계 요약:
  1. rowspan/colspan 을 직사각 그리드로 전개 (병합 셀 값 복제)
  2. 다단 헤더를 컬럼 라벨로 합성 ("2024년 > 3분기")
  3. 행 단위 KV 선형화 -> 임베딩 텍스트 / 마크다운 표 -> LLM 컨텍스트 (이중 표현)
  4. 청크 경계는 절대 행을 쪼개지 않음

의존성: beautifulsoup4 + html5lib. lxml 불필요.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, NavigableString, Tag

SKIP = {"script", "style", "noscript", "head", "svg", "template"}
HEAD = {"h1", "h2", "h3", "h4", "h5", "h6"}
BLOCK = {"p", "div", "li", "blockquote", "pre", "section", "article",
         "figcaption", "dd", "dt", "address", *HEAD}

# html.parser 는 HTML5 "in table" 삽입 모드를 구현하지 않아 안 닫힌 <td>/<tr> 을
# 자동으로 닫지 않고 중첩시킨다. OCR 엔진 산출물이 정확히 그 형태라, 그 백엔드로는
# 표가 조용히 폭발한다 (fixtures/03: 5열 표가 53열로). html5lib 은 실제 HTML5
# 트리 구성 알고리즘을 구현하므로 올바르게 닫는다. ~10배 느리지만 수집은 1회뿐이다.
BACKEND = "html5lib"

_WS = re.compile(r"[\s ​‌‍﻿]+")
# ponytail: 한국어 문장 종결 휴리스틱. 인용부호/괄호 안 마침표는 오분할 가능 —
# 오분할이 검색 품질에 실측으로 걸리면 kiwipiepy 문장분리로 교체.
_SENT = re.compile(r"(?<=[.!?。？！])\s+|(?<=[다요음임함])\.\s+")


def clean(s: str) -> str:
    """공백 정규화 + 유니코드 NFKC.

    NFD(자모 분리) 유입은 한국어에서 치명적이다. OCR/HTML/macOS 를 거친 텍스트가
    NFD 로 들어오면 '한글' != '한글' 이 되어 임베딩·정확일치가 조용히 망가진다.
    눈으로는 구분되지 않는다.

    NFC 가 아니라 NFKC 인 이유: NFC 는 전각을 접지 않아 '４,３２０' != '4,320' 이
    남는다. 한국어 문서에는 전각 숫자, ㈜, ①, ㎡, ℃ 가 흔하고 전부 어휘 검색을
    깨뜨린다. NFKC 는 이들을 '4,320', '(주)', '1', 'm2', '°C' 로 접는다.

    !! 질의에도 같은 함수를 적용할 것 !! 색인만 정규화하고 질의를 안 하면
    오히려 매칭이 더 나빠진다. 중요한 건 양쪽이 같은 형태라는 것이다.
    """
    return _WS.sub(" ", unicodedata.normalize("NFKC", s)).strip()


# --------------------------------------------------------------------------- #
# 표 정규화
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Cell:
    text: str
    header: bool
    origin: tuple[int, int]  # 병합 영역의 좌상단. 복제된 셀 판별에 사용


@dataclass
class Table:
    grid: list[list[Cell]]
    labels: list[str]       # 컬럼 라벨 (다단 헤더 합성 결과)
    n_header: int           # 헤더로 소비한 행 수
    caption: str = ""
    n_src_cells: int = 0    # 원본 <td>/<th> 개수 (불변식 검증용)

    @property
    def body(self) -> list[list[Cell]]:
        return self.grid[self.n_header:]


def _own(tag: Tag, table: Tag) -> bool:
    """중첩 표의 요소를 부모 표 것으로 오인하지 않게 하는 가드."""
    return tag.find_parent("table") is table


def build_table(table: Tag) -> Table:
    """rowspan/colspan 을 전개해 직사각 그리드로 만든다."""
    occupied: dict[tuple[int, int], Cell] = {}
    n_src = 0
    rows = [tr for tr in table.find_all("tr") if _own(tr, table)]

    for r, tr in enumerate(rows):
        c = 0
        for td in tr.find_all(["td", "th"]):
            if not _own(td, table):
                continue
            n_src += 1
            while (r, c) in occupied:
                c += 1
            rs = max(1, _int(td.get("rowspan"), 1))
            cs = max(1, _int(td.get("colspan"), 1))
            # ponytail: 중첩 표는 부모 셀 안에서 평문으로 눌러 담는다.
            # 중첩이 의미를 갖는 문서가 나오면 재귀 파싱으로 승격.
            cell = Cell(clean(td.get_text(" ", strip=True)), td.name == "th", (r, c))
            for dr in range(rs):
                for dc in range(cs):
                    occupied[(r + dr, c + dc)] = cell
            c += cs

    if not occupied:
        return Table([], [], 0, n_src_cells=n_src)

    n_rows = max(r for r, _ in occupied) + 1
    n_cols = max(c for _, c in occupied) + 1
    blank = Cell("", False, (-1, -1))
    grid = [[occupied.get((r, c), blank) for c in range(n_cols)] for r in range(n_rows)]

    n_header = _header_rows(grid)
    cap = table.find("caption")
    return Table(
        grid=grid,
        labels=_labels(grid, n_header, n_cols),
        n_header=n_header,
        caption=clean(cap.get_text(" ", strip=True)) if cap and _own(cap, table) else "",
        n_src_cells=n_src,
    )


def _int(v, default: int) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


_DIGIT = re.compile(r"\d")


def _header_rows(grid: list[list[Cell]]) -> int:
    """헤더로 쓸 선행 행 수.

    <th> 가 있으면 그걸 믿는다. 없으면 '숫자가 없는 선행 행' 을 헤더로 본다 —
    Word/한글 내보내기와 OCR 산출물은 <th> 를 전혀 쓰지 않아 태그로는 다단 헤더를
    찾을 수 없고, 헤더 행은 값(금액·연도·수량)을 담지 않는다는 성질을 이용한다.

    ponytail: 헤더에 연도가 들어간 th 없는 표('구분|2024년|2025년')는 1행으로 떨어진다
    (= 종전 동작, 회귀 아님). 다단 헤더까지 필요해지면 배경색/굵기 신호를 추가할 것.
    """
    if any(c.header for row in grid for c in row):
        n = 0
        for row in grid:
            if row and all(c.header for c in row):
                n += 1
            else:
                break
    else:
        n = 0
        for row in grid[:3]:                      # 4단 이상 헤더는 실물에 거의 없다
            if row and not any(_DIGIT.search(c.text) for c in row):
                n += 1
            else:
                break
    return min(max(n, 1), max(0, len(grid) - 1))


def _labels(grid, n_header: int, n_cols: int) -> list[str]:
    """다단 헤더를 컬럼별로 합성. 상위>하위 순, 중복 제거."""
    out = []
    for c in range(n_cols):
        parts: list[str] = []
        for r in range(n_header):
            t = grid[r][c].text
            if t and t not in parts:
                parts.append(t)
        out.append(" > ".join(parts))
    return out


def row_kv(labels: list[str], row: list[Cell]) -> str:
    """행 -> "라벨: 값" 선형화. colspan 으로 복제된 셀은 1회만."""
    parts, prev = [], None
    for lab, cell in zip(labels, row):
        if not cell.text or cell.origin == prev:
            prev = cell.origin
            continue
        prev = cell.origin
        parts.append(f"{lab}: {cell.text}" if lab else cell.text)
    return ", ".join(parts)


def to_markdown(labels: list[str], rows: list[list[Cell]]) -> str:
    """LLM 컨텍스트용. 병합 셀은 값을 반복해 명시 (LLM 이 읽기 쉬움)."""
    head = "| " + " | ".join(labels) + " |"
    sep = "| " + " | ".join("---" for _ in labels) + " |"
    body = ["| " + " | ".join(c.text for c in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


# --------------------------------------------------------------------------- #
# 문서 순회 + 청킹
# --------------------------------------------------------------------------- #

@dataclass
class Chunk:
    text: str                       # 임베딩 대상
    context: str                    # LLM 에 넘길 원형 (표는 마크다운)
    kind: str                       # "text" | "table"
    heading: str = ""               # 상위 섹션 경로
    table_id: int = -1
    row_span: tuple[int, int] = (-1, -1)  # body 기준 [start, end)
    oversize: bool = False

    def as_dict(self) -> dict:
        return {**self.__dict__}


@dataclass
class _Walk:
    blocks: list = field(default_factory=list)
    trail: list[str] = field(default_factory=lambda: [""] * 6)

    def heading(self) -> str:
        return " > ".join(h for h in self.trail if h)


def _walk(node, w: _Walk):
    for child in getattr(node, "children", []):
        if isinstance(child, NavigableString):
            t = clean(str(child))
            if t:
                w.blocks.append(("text", t, w.heading()))
            continue
        if not isinstance(child, Tag) or child.name in SKIP:
            continue
        if child.name == "table":
            w.blocks.append(("table", child, w.heading()))
            continue          # 표 내부로는 내려가지 않는다
        if child.name in HEAD:
            lvl = int(child.name[1]) - 1
            w.trail[lvl] = clean(child.get_text(" ", strip=True))
            for i in range(lvl + 1, 6):
                w.trail[i] = ""
            w.blocks.append(("break", "", w.heading()))
            continue  # 제목 자체는 본문에 넣지 않는다. 모든 청크의 prefix 로 붙는다.
        if child.name == "br":
            w.blocks.append(("break", "", w.heading()))
            continue
        _walk(child, w)
        if child.name in BLOCK:
            w.blocks.append(("break", "", w.heading()))


def parse(html: str, max_chars: int = 900, backend: str = BACKEND) -> list[Chunk]:
    """HTML -> 청크 목록 (문서 순서 유지).

    max_chars 는 글자 기준. BGE-M3 기준 한국어 1글자 ≈ 0.6~0.9 토큰이므로
    900자 ≈ 550~800 토큰. 모델 상한에 맞춰 조정할 것.
    backend 는 BACKEND 참조. 속도가 급하고 입력이 깨끗하다고 확신하면
    backend="html.parser" 로 낮출 수 있다 (표가 깨질 위험을 감수하는 것).
    """
    soup = BeautifulSoup(html, backend)
    w = _Walk()
    _walk(soup.body or soup, w)

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_head = ""
    tid = 0

    def flush():
        nonlocal buf, buf_head
        if buf:
            for piece in _pack_text(" ".join(buf), max_chars):
                body = f"{buf_head}\n{piece}" if buf_head else piece
                chunks.append(Chunk(text=body, context=body, kind="text", heading=buf_head))
            buf = []

    for kind, payload, heading in w.blocks:
        if kind == "break":
            continue
        if kind == "text":
            if heading != buf_head:
                flush()
                buf_head = heading
            buf.append(payload)
            continue
        flush()
        buf_head = heading
        chunks.extend(_pack_table(build_table(payload), tid, heading, max_chars))
        tid += 1
    flush()
    return [c for c in chunks if c.text.strip()]


def _pack_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []
    out, cur = [], ""
    for sent in _SENT.split(text):
        if not sent:
            continue
        if cur and len(cur) + len(sent) + 1 > max_chars:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    return out


def is_prose_row(row: list[Cell], min_chars: int = 60) -> bool:
    """표 전체 폭을 한 셀이 덮고 내용이 긴 행 = 데이터가 아니라 본문 안내문.

    한국어 실무 문서(Word/한글 내보내기)에서 매우 흔하다 — 표 맨 아래에 ※ 주석이나
    처리 안내를 한 칸짜리 행으로 붙인다. 이걸 데이터 행으로 다루면
    '구분: ※ 요금제 선택을...' 같은 쓰레기 KV 가 나온다.
    """
    if len(row) < 2:
        return False
    origins = {c.origin for c in row}
    return len(origins) == 1 and origins != {(-1, -1)} and len(row[0].text) >= min_chars


def _pack_table(t: Table, tid: int, heading: str, max_chars: int) -> list[Chunk]:
    """행 원자적 청킹: 헤더 문맥을 매 청크에 반복하고 행은 절대 쪼개지 않는다."""
    if not t.body:
        return []
    title = " / ".join(x for x in (heading, t.caption) if x)
    prefix = f"[표] {title}" if title else "[표]"
    cols = ", ".join(l for l in t.labels if l)
    header_line = f"{prefix}\n컬럼: {cols}" if cols else prefix

    out, cur, start = [], [], 0
    budget = max_chars - len(header_line)

    def emit(end: int):
        rows = t.body[start:end]
        lines = [row_kv(t.labels, r) for r in rows]
        body = "\n".join(lines)
        out.append(Chunk(
            text=f"{header_line}\n{body}",
            context=f"{prefix}\n{to_markdown(t.labels, rows)}",
            kind="table", heading=heading, table_id=tid, row_span=(start, end),
            oversize=any(len(l) > budget for l in lines),
        ))

    used = 0
    for i, row in enumerate(t.body):
        if is_prose_row(row):
            # 표를 여기서 끊고 안내문은 본문 청크로 따로 낸다
            if cur:
                emit(i)
                cur, used = [], 0
            txt = row[0].text
            for piece in _pack_text(txt, max_chars):
                body = f"{prefix}\n{piece}" if title else piece
                out.append(Chunk(text=body, context=body, kind="text",
                                 heading=heading, table_id=tid, row_span=(i, i + 1)))
            start = i + 1
            continue
        line = row_kv(t.labels, row)
        if not line:
            continue
        if cur and used + len(line) + 1 > budget:
            emit(i)
            cur, used, start = [], 0, i
        cur.append(line)
        used += len(line) + 1
    if cur:
        emit(len(t.body))
    return out


# --------------------------------------------------------------------------- #
# 자체 검증: 라벨 없이 돌아가는 구조 불변식
# --------------------------------------------------------------------------- #

def invariants(html: str, chunks: list[Chunk] | None = None, **kw) -> dict:
    """정답 라벨 0개로 파서 회귀를 잡는 검사. 모든 문서에 공짜로 돌릴 수 있다."""
    chunks = parse(html, **kw) if chunks is None else chunks
    soup = BeautifulSoup(html, kw.get("backend", BACKEND))
    tags = [t for t in soup.find_all("table") if t.find_parent("table") is None]
    tables = [build_table(t) for t in tags]

    rect = all(len({len(r) for r in t.grid}) <= 1 for t in tables if t.grid)
    # 원본 셀이 차지해야 할 칸 수 == 실제로 차지한 칸 수.
    # 들쭉날쭉한 행을 메우는 빈 칸(origin == (-1,-1))은 정상이므로 제외한다 —
    # 이게 잡아야 하는 건 병합 영역이 겹쳐 셀이 덮여 사라지는 경우다.
    conserved = all(
        sum(1 for r in t.grid for c in r if c.origin != (-1, -1)) == sum(
            max(1, _int(td.get("rowspan"), 1)) * max(1, _int(td.get("colspan"), 1))
            for td in tag.find_all(["td", "th"]) if _own(td, tag)
        )
        for tag, t in zip(tags, tables) if t.grid
    )
    # 텍스트 보존율: 원문 토큰 중 어떤 청크에도 안 나타난 비율
    blob = " ".join(c.text for c in chunks)
    src = [w for w in _WS.split(clean((soup.body or soup).get_text(" "))) if w]
    missing = [w for w in set(src) if w not in blob]
    # 행 원자성: 모든 body 행의 KV 가 정확히 한 청크 안에 통째로 들어있는가
    atomic = True
    for tid, t in enumerate(tables):
        owned = [c for c in chunks if c.table_id == tid]
        for r in t.body:
            if is_prose_row(r):          # 본문 청크로 빠진 행. KV 로 존재하지 않는 게 정상
                continue
            kv = row_kv(t.labels, r)
            if kv and not any(kv in c.text for c in owned):
                atomic = False
    return {
        "rectangular": rect,
        "cell_conservation": conserved,
        "text_coverage": 1.0 - len(missing) / max(1, len(set(src))),
        "missing_tokens": missing[:10],
        "row_atomic": atomic,
        "n_chunks": len(chunks),
        "n_tables": len(tables),
        "oversize_chunks": sum(c.oversize for c in chunks),
    }




FIXTURE = """
<h1>2024 사업보고서</h1>
<h2>3. 재무 현황</h2>
<p>당사의 2024년 실적은 전년 대비 개선되었습니다. 세부 내역은 아래 표와 같습니다.</p>
<table>
  <caption>부문별 매출 (단위: 억원)</caption>
  <tr><th rowspan="2">사업부문</th><th colspan="2">2024년</th><th rowspan="2">증감률</th></tr>
  <tr><th>상반기</th><th>하반기</th></tr>
  <tr><td rowspan="2">반도체</td><td>1,200</td><td>1,450</td><td>20.8%</td></tr>
  <tr><td>820</td><td>910</td><td>11.0%</td></tr>
  <tr><td>디스플레이</td><td>640</td><td>588</td><td>-8.1%</td></tr>
</table>
<p>반도체 부문의 성장이 전체 실적을 견인하였습니다.</p>
"""

# OCR 산출물은 깨져서 온다. 파싱은 신뢰 경계이므로 죽으면 안 된다.
BROKEN = """
<div><p>서론 문단<table>
<tr><td colspan="9">전체 폭을 넘는 colspan</td></tr>
<tr><td>A</td><td rowspan="0">rowspan 0</td><td></td>
<tr><td>B<td>닫는 태그 없음<td>세번째
<tr><td>중첩<table><tr><td>안쪽1</td><td>안쪽2</td></tr></table></td><td>C</td></tr>
<tr><td rowspan="abc">숫자 아닌 span</td><td>D</td></tr>
</table>
<p>결론 문단</p></div>
"""

if __name__ == "__main__":
    # 1) 깨진 마크업 내성: 죽지 않고, 그리드는 여전히 직사각이어야 한다
    bc = parse(BROKEN)
    binv = invariants(BROKEN)
    assert binv["rectangular"], "깨진 입력에서 그리드가 무너짐"
    assert bc, "깨진 입력에서 청크가 하나도 안 나옴"
    bt = build_table(BeautifulSoup(BROKEN, BACKEND).find("table"))
    assert all(len(r) == len(bt.grid[0]) for r in bt.grid)
    assert "안쪽1" in " ".join(c.text for r in bt.grid for c in r), "중첩 표 텍스트 유실"
    assert binv["text_coverage"] > 0.98, binv["missing_tokens"]

    ch = parse(FIXTURE, max_chars=400)
    inv = invariants(FIXTURE, ch, max_chars=400)

    assert inv["rectangular"], "그리드가 직사각이 아님"
    assert inv["cell_conservation"], "셀 수 불일치 (rowspan/colspan 전개 오류)"
    assert inv["row_atomic"], "행이 청크 경계에서 쪼개짐"
    assert inv["text_coverage"] > 0.98, f"텍스트 유실: {inv['missing_tokens']}"

    t = build_table(BeautifulSoup(FIXTURE, BACKEND).find("table"))
    assert t.n_header == 2, t.n_header
    assert t.labels == ["사업부문", "2024년 > 상반기", "2024년 > 하반기", "증감률"], t.labels
    # rowspan 전개: 2번째 반도체 행에도 '반도체'가 채워져 있어야 검색이 된다
    assert t.body[1][0].text == "반도체"
    assert "사업부문: 반도체" in row_kv(t.labels, t.body[1])
    assert "2024년 > 하반기: 910" in row_kv(t.labels, t.body[1])
    # colspan 으로 복제된 셀은 KV 에서 1회만 ('2024년' 이 2칸을 덮고 있음)
    assert t.grid[0][1].origin == t.grid[0][2].origin
    assert row_kv(t.labels, t.grid[0]).count("2024년 > 하반기") == 0

    print("OK", {k: v for k, v in inv.items() if k != "missing_tokens"})
    for c in ch:
        print(f"\n--- {c.kind} rows={c.row_span} ---\n{c.text}")
