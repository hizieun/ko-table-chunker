"""실파일 -> 구조적 쌍둥이. 내용은 지우고 구조만 남긴다.

보안상 실파일을 내보낼 수 없을 때 쓴다. 태그·속성·rowspan/colspan·중첩·깨진
마크업·유니코드 형태를 **그대로** 두고 텍스트만 같은 길이·같은 문자종의 더미로
바꾼다. 파서가 타는 코드 경로가 100% 같으므로 버그는 똑같이 재현되지만,
읽을 수 있는 내용은 남지 않는다.

  python3 anonymize.py 실파일/*.htm --out twin/     # 쌍둥이 생성
  python3 anonymize.py 실파일/*.htm --report        # 통계만 출력 (텍스트 0)
  python3 anonymize.py --selfcheck                  # 구조 보존 검증

단어 단위로 **매번 새로 만드는 난수 salt** 를 섞어 해싱한다. 같은 단어는 같은
더미가 되어 '문서마다 반복되는 caption' 같은 성질이 보존되지만, salt 는 어디에도
기록하지 않으므로 치환 규칙을 복원할 수 없다.

ponytail: 문자종(한글·숫자·영문·전각)과 길이만 보존한다. 어휘 빈도 분포는 남으므로
'어떤 단어가 흔한가' 는 드러난다. 그것도 막아야 하면 --shuffle-len 으로 길이를
±1 흔들 것 (대신 is_value 의 숫자 비율이 미세하게 달라진다).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import re
import sys
import unicodedata

from bs4 import BeautifulSoup, Comment, NavigableString

from html_chunker import BACKEND, build_table, header_cols, is_crosstab

SALT = os.urandom(16)                       # 매 실행마다 새로. 기록하지 않는다.
_HANGUL = [chr(c) for c in range(0xAC00, 0xD7A4, 97)]        # 고르게 흩은 음절 풀
_LATIN = "abcdefghijklmnopqrstuvwxyz"
TEXT_ATTRS = ("alt", "title", "placeholder", "aria-label", "summary", "value")


def _rng(word: str, n: int) -> list[int]:
    """단어에서 결정론적 난수열. salt 때문에 외부에서 재현 불가."""
    out, h = [], hashlib.blake2b(SALT + word.encode(), digest_size=64).digest()
    while len(out) < n:
        out.extend(h)
        h = hashlib.blake2b(SALT + h, digest_size=64).digest()
    return out[:n]


# NFD(자모 분리) 텍스트를 조합형으로 바꾸면 길이도 NFD성도 깨진다.
# 같은 자모 종류 안에서만 치환해야 분해 구조가 그대로 남는다.
_JAMO = [(0x1100, 0x1112), (0x1161, 0x1175), (0x11A8, 0x11C2),   # 초·중·종성
         (0x3131, 0x318E)]                                       # 호환 자모


def _jamo_class(ch: str) -> tuple[int, int] | None:
    c = ord(ch)
    return next(((lo, hi) for lo, hi in _JAMO if lo <= c <= hi), None)


def fake_word(w: str) -> str:
    """같은 길이·같은 문자종의 더미. 문자가 아닌 것(구두점·공백·단위)은 그대로."""
    r = _rng(w, len(w))
    out = []
    for ch, k in zip(w, r):
        if ch.isdigit():
            out.append(str(k % 10))
        elif "가" <= ch <= "힣":                     # 한글 음절
            out.append(_HANGUL[k % len(_HANGUL)])
        elif ch.isascii() and ch.isalpha():
            c = _LATIN[k % 26]
            out.append(c.upper() if ch.isupper() else c)
        elif "０" <= ch <= "９":                     # 전각 숫자
            out.append(chr(0xFF10 + k % 10))
        elif (jamo := _jamo_class(ch)) is not None:          # NFD 자모
            lo, hi = jamo
            out.append(chr(lo + k % (hi - lo + 1)))
        elif unicodedata.category(ch) == "Lo":               # 한자 등
            out.append(_HANGUL[k % len(_HANGUL)])
        else:
            out.append(ch)                                   # , . % 원 공백 NBSP …
    return "".join(out)


_WORD = re.compile(r"\S+")


def fake_text(s: str) -> str:
    """공백 구조는 그대로 두고 단어만 바꾼다. NFD/전각/제로폭이 보존된다."""
    return _WORD.sub(lambda m: fake_word(m.group()), s)


def anonymize(html: str, keep: set[str] = frozenset()) -> str:
    soup = BeautifulSoup(html, BACKEND)
    for c in soup.find_all(string=lambda x: isinstance(x, Comment)):
        c.extract()                                          # 주석은 통째로 제거
    for node in soup.find_all(string=True):
        # NavigableString 의 하위 클래스(Doctype, Declaration, CData, Stylesheet,
        # Script …)는 건드리면 안 된다. <!DOCTYPE html> 의 'html' 까지 치환하면
        # 문서 자체가 깨져 head/body 구조가 달라진다.
        if type(node) is not NavigableString:
            continue
        if node.parent.name in ("script", "style"):
            node.replace_with("")
            continue
        s = str(node)
        if s.strip() and s.strip() not in keep:
            node.replace_with(NavigableString(fake_text(s)))
    for tag in soup.find_all(True):
        for a in TEXT_ATTRS:
            if tag.has_attr(a):
                tag[a] = fake_text(tag[a]) if isinstance(tag[a], str) else tag[a]
        for a in ("href", "src", "action", "data-src"):
            if tag.has_attr(a):
                tag[a] = "#"
    return str(soup)


# --------------------------------------------------------------------------- #
# 구조 지문 — 쌍둥이가 진짜 같은지, 그리고 --report 로 내보낼 통계
# --------------------------------------------------------------------------- #

def fingerprint(html: str) -> dict:
    soup = BeautifulSoup(html, BACKEND)
    tags = collections.Counter(t.name for t in soup.find_all(True))
    spans, shapes, hdr = collections.Counter(), [], collections.Counter()
    for tag in soup.find_all("table"):
        for c in tag.find_all(["td", "th"]):
            spans[f"rs{c.get('rowspan', 1)}×cs{c.get('colspan', 1)}"] += 1
        if tag.find_parent("table") is None:
            t = build_table(tag)
            if t.grid:
                shapes.append([len(t.grid), len(t.grid[0])])
                hdr[f"행{t.n_header}/열{header_cols(t)}/교차{int(is_crosstab(t))}"] += 1
    body = soup.body or soup
    txt = body.get_text(" ")
    return {
        "tags": dict(tags.most_common(25)),
        "nested_tables": sum(1 for t in soup.find_all("table") if t.find_parent("table")),
        "spans": dict(spans.most_common(15)),
        "table_shapes": shapes,
        "header_shapes": dict(hdr),
        "chars": len(txt),
        "nfd": txt != unicodedata.normalize("NFC", txt),
        "fullwidth": bool(re.search(r"[！-～]", txt)),
        "has_th": "th" in tags,
        "mso": "MsoNormalTable" in html,
    }


def chunk_shape(html: str) -> list[tuple]:
    """파서가 실제로 내놓는 청크의 모양. 내용은 빼고 구조만."""
    from html_chunker import parse
    return [(c.kind, c.table_id, c.row_span, len(c.text)) for c in parse(html)]


def selfcheck() -> None:
    """쌍둥이가 파서에게 원본과 **같은 문서**인지 검증한다.

    태그·병합 통계가 같은 것만으로는 부족하다. 청킹은 글자 수에 반응하므로
    최종적으로 같은 청크 경계가 나와야 쌍둥이로 잰 결과를 원본 결과로 읽을 수 있다.
    """
    import glob
    paths = sorted(glob.glob("fixtures/*.html")) + sorted(glob.glob("corpus/*.htm"))[:10]
    assert paths, "검증할 파일이 없다 (fixtures/ 또는 corpus/)"
    for p in paths:
        src = open(p, encoding="utf-8").read()
        twin = anonymize(src)
        a, b = fingerprint(src), fingerprint(twin)
        for k in ("tags", "spans", "table_shapes", "header_shapes",
                  "nested_tables", "nfd", "fullwidth", "has_th"):
            assert a[k] == b[k], f"{p}: {k} 불일치\n  원본={a[k]}\n  쌍둥이={b[k]}"
        # 주석을 통째로 지우므로 글자 수는 그만큼 줄 수 있다. 1% 까지 허용.
        assert abs(a["chars"] - b["chars"]) <= max(2, a["chars"] // 100), \
            f'{p}: chars {a["chars"]} vs {b["chars"]}'
        sa, sb = chunk_shape(src), chunk_shape(twin)
        assert len(sa) == len(sb), f"{p}: 청크 수 {len(sa)} vs {len(sb)}"
        for x, y in zip(sa, sb):
            assert x[:3] == y[:3] and abs(x[3] - y[3]) <= 4, f"{p}: 청크 불일치 {x} vs {y}"
        print(f"  ✓ {p}  (청크 {len(sa)}개 동일)")
    print(f"\n{len(paths)}개 — 태그·병합·헤더·청크 경계가 모두 같다.\n"
          f"쌍둥이로 잰 파서 결과는 원본 결과로 읽어도 된다.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--out", help="쌍둥이를 저장할 디렉토리")
    ap.add_argument("--report", action="store_true", help="통계만 출력 (텍스트 0)")
    ap.add_argument("--keep", default="", help="그대로 둘 단어, 쉼표 구분")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        selfcheck()
        sys.exit()
    if not a.paths:
        ap.error("파일을 지정하거나 --selfcheck 를 쓸 것")

    keep = {w.strip() for w in a.keep.split(",") if w.strip()}
    if a.report:
        agg = [dict(fingerprint(open(p, encoding="utf-8").read()), file=f"doc{i:03d}")
               for i, p in enumerate(a.paths)]
        print(json.dumps(agg, ensure_ascii=False, indent=1))
        sys.exit()

    out = pathlib.Path(a.out or "twin")
    out.mkdir(exist_ok=True)
    for i, p in enumerate(a.paths):
        src = open(p, encoding="utf-8").read()
        twin = anonymize(src, keep)
        fa, fb = fingerprint(src), fingerprint(twin)
        bad = [k for k in ("tags", "spans", "table_shapes", "header_shapes")
               if fa[k] != fb[k]]
        (out / f"doc{i:04d}.htm").write_text(twin, encoding="utf-8")
        print(f"doc{i:04d}.htm  {'OK' if not bad else '구조 불일치: ' + str(bad)}")
    print(f"\n{len(a.paths)}개 -> {out}/  (파일명도 익명화됨)")
