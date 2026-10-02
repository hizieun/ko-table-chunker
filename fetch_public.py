"""공개된 한국어 실문서를 받아 테스트 코퍼스로 쓴다.

고객사 문서를 반출할 수 없을 때의 대안. 합성 코퍼스는 **내가 상상한 패턴만**
담지만, 전자공시(DART) 사업보고서는 구조적으로 업무 문서 그 자체다 — 병합 표,
다단 헤더, 레이아웃용 표, 깨진 마크업이 실제 비율로 들어있다. 공개 자료라
저장·공유에 제약이 없다.

  python3 fetch_public.py --n 12 --out public/
  python3 evaluate.py --l1 public/*.html

ponytail: DART 하나만 받는다. 출처를 늘리는 건 방언이 더 필요해질 때.
서버에 부담 주지 않도록 문서 사이에 쉬고, 기본 12건으로 제한한다.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import time
import urllib.request

# HTTP 헤더는 latin-1 만 담는다. 한글을 넣으면 UnicodeEncodeError.
UA = {"User-Agent": "Mozilla/5.0 (ko-table-chunker parser test)",
      "Referer": "https://dart.fss.or.kr/"}
LIST = ("https://dart.fss.or.kr/dsab007/detailSearch.ax?currentPage=1"
        "&maxResults={n}&finalReport=recent&businessCode=all&publicType={t}")
VIEWDOC = re.compile(r'viewDoc\("(\d+)",\s*"(\d+)",\s*"([^"]*)",\s*"([^"]*)",'
                     r'\s*"([^"]*)",\s*"([^"]*)"')


def _get(url: str, timeout: int = 30) -> bytes:
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=timeout).read()


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "euc-kr", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def rcp_numbers(n: int, kind: str) -> list[str]:
    html = _decode(_get(LIST.format(n=n * 2, t=kind)))
    seen = dict.fromkeys(re.findall(r"rcpNo=(\d{14})", html)
                         or re.findall(r"openReportViewer\('(\d{14})'", html))
    return list(seen)[:n]


def fetch_doc(rcp: str) -> str | None:
    """공시 뷰어의 실제 본문 HTML. 못 찾으면 None."""
    m = VIEWDOC.search(_decode(_get(f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcp}")))
    if not m:
        return None
    r, d, ele, off, ln, dtd = m.groups()
    return _decode(_get(f"https://dart.fss.or.kr/report/viewer.do?rcpNo={r}"
                        f"&dcmNo={d}&eleId={ele}&offset={off}&length={ln}&dtd={dtd}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="받을 문서 수 (최대 30)")
    ap.add_argument("--out", default="public")
    ap.add_argument("--kind", default="A001",
                    help="A001 사업보고서 / A002 반기 / A003 분기")
    ap.add_argument("--sleep", type=float, default=1.5, help="문서 간 대기(초)")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(exist_ok=True)
    got = 0
    for i, rcp in enumerate(rcp_numbers(min(a.n, 30), a.kind)):
        try:
            doc = fetch_doc(rcp)
        except Exception as e:
            print(f"  ! {rcp}: {type(e).__name__}")
            continue
        if not doc:
            print(f"  - {rcp}: 본문 없음")
            continue
        (out / f"{rcp}.html").write_text(doc, encoding="utf-8")
        got += 1
        print(f"  {rcp}  {len(doc):>8,}자  표 {doc.lower().count('<table'):>3}개")
        time.sleep(a.sleep)
    print(f"\n{got}개 -> {out}/   (공개 전자공시. 고객사 데이터 아님)")
