"""실파일급 합성 코퍼스 생성기.

고객사 실파일(CONTS0039.htm)의 구조적 특성을 따른다:
  - CMS 콘텐츠 ID 파일명, 파일당 표 1~3개 + 본문
  - Word/한글 내보내기(MsoNormalTable, <th> 없음, 인라인 style 폭탄)
  - 증권/금융 업무 도메인

손으로 만든 fixtures/ 5종이 못 잡는 걸 찾는 게 목적이므로 표 원형을 7가지로
나누고 변주(th 유무, caption 유무, 안내문 행, 각주, 빈칸, 전각, NFD)를 섞는다.

실행: python3 make_corpus.py [--n 40] [--out corpus] [--seed 0]
"""

from __future__ import annotations

import argparse
import pathlib
import random
import unicodedata

CORPS = ["가온증권", "한빛투자", "대명자산운용", "서진캐피탈", "누리저축은행",
         "명진생명", "태성화재", "율촌은행", "동방증권", "선진자산", "다산투자",
         "해성금융", "청람증권", "백산캐피탈", "우진자산운용"]
INST = ["증권결제원", "금융정보원", "한국예탁결제", "전자인증원", "금융결제망"]
SEGS = ["국내주식", "해외주식", "채권", "펀드", "ELS/DLS", "연금저축", "ISA",
        "CMA", "신용융자", "대주거래", "외환", "파생상품"]
BRANCH = ["강남", "여의도", "판교", "부산", "대구", "광주", "대전", "인천",
          "울산", "수원", "청주", "전주"]
GRADE = ["일반", "우대", "VIP", "VVIP"]

STY_HDR = ('style="background: rgb(238, 245, 254); padding: 0cm 5.4pt; '
           'border: 1pt solid rgb(210, 228, 251); border-image: none; '
           'vertical-align: middle;" bgcolor="#eef5fe"')
STY_CEL = ('style="padding: 0cm 5.4pt; border: 1pt solid rgb(210, 228, 251); '
           'border-image: none; vertical-align: middle; '
           'background-color: rgb(249, 249, 249);"')


def cell(txt, *, th=False, rs=1, cs=1, hdr=False, w=None, bold=False):
    tag = "th" if th else "td"
    a = [f'width="{w}"' if w else "", f'rowspan="{rs}"' if rs > 1 else "",
         f'colspan="{cs}"' if cs > 1 else "", 'valign="middle"',
         STY_HDR if hdr else STY_CEL]
    inner = f'<span style="font-weight:700">{txt}</span>' if bold else f"<span>{txt}</span>"
    return (f'<{tag} {" ".join(x for x in a if x)}>'
            f'<p class="MsoNormal">{inner}<o:p></o:p></p></{tag}>')


def table(rows_html, caption=None, width=735):
    cap = f"<caption>{caption}</caption>" if caption else ""
    return (f'<table width="{width}" class="MsoNormalTable" border="1" cellspacing="0" '
            f'cellpadding="0" style="border-collapse: collapse;">{cap}<tbody>'
            + "".join(rows_html) + "</tbody></table>")


def money(r, lo=1, hi=900):
    return f"{r.randint(lo, hi) * 100:,}원"


# --------------------------------------------------------------------------- #
# 표 원형 7가지
# --------------------------------------------------------------------------- #

def t_crosstab(r):
    """기관 x (개인/법인) 교차표. 실파일과 같은 형태. 2단 헤더 + 좌측 2단 분류."""
    a, b = r.sample(INST, 2)
    th = r.random() < 0.2
    rows = [
        "<tr>" + cell("구  분", th=th, rs=2, cs=2, hdr=True, bold=True)
        + cell(a, th=th, cs=2, hdr=True, bold=True)
        + cell(b, th=th, cs=2, hdr=True, bold=True) + "</tr>",
        "<tr>" + "".join(cell(x, th=th, hdr=True) for x in ("개인", "법인", "개인", "법인")) + "</tr>",
    ]
    for kind in ("범용", "전용"):
        rows.append("<tr>" + cell(kind, rs=2, hdr=True, bold=True)
                    + cell("수수료", hdr=True, bold=True)
                    + "".join(cell(x) for x in
                              (money(r, 30, 60) if kind == "범용" else "-",
                               money(r, 600, 1200), money(r, 30, 60) if kind == "범용" else "-",
                               money(r, 400, 1100))) + "</tr>")
        rows.append("<tr>" + cell("이용범위", hdr=True, bold=True)
                    + cell(f"전 {r.choice(['제휴기관', '계열사', '가맹점'])}/단체", cs=2)
                    + cell(f"전 {r.choice(['은행', '증권사', '보험사'])}, 전자정부", cs=2) + "</tr>")
    return table(rows, f"인증서 종류별 수수료 ({r.choice(['부가세 포함', '연간 기준'])})")


def t_period(r):
    """기간 x 분기 교차표. 좌측은 사업부문 계층."""
    y = r.randint(2019, 2025)
    rows = ["<tr>" + cell("사업부문", rs=2, cs=2, hdr=True, bold=True)
            + cell(f"{y}년", cs=2, hdr=True, bold=True)
            + cell(f"{y - 1}년", cs=2, hdr=True, bold=True) + "</tr>",
            "<tr>" + "".join(cell(x, hdr=True) for x in ("상반기", "하반기") * 2) + "</tr>"]
    for seg in r.sample(SEGS, r.randint(3, 5)):
        rows.append("<tr>" + cell(seg, rs=2, hdr=True, bold=True) + cell("수익", hdr=True)
                    + "".join(cell(f"{r.randint(100, 9000):,}") for _ in range(4)) + "</tr>")
        rows.append("<tr>" + cell("비중", hdr=True)
                    + "".join(cell(f"{r.uniform(0.5, 40):.1f}%") for _ in range(4)) + "</tr>")
    return table(rows, f"{y}년 부문별 손익 (단위: 백만원)")


def t_records(r):
    """행 = 레코드인 단순 목록. 교차표로 오판하면 안 된다."""
    n = r.randint(4, 14)
    rows = ["<tr>" + "".join(cell(x, hdr=True, bold=True) for x in
                             ("지점명", "소재지", "등급", "관리자산", "고객수")) + "</tr>"]
    for br in r.sample(BRANCH, min(n, len(BRANCH))):
        rows.append("<tr>" + "".join(cell(x) for x in (
            f"{br}지점", f"{br} 일원", r.choice(GRADE),
            f"{r.randint(100, 9999):,}", f"{r.randint(50, 5000):,}")) + "</tr>")
    return table(rows, "지점 현황" if r.random() < 0.7 else None)


def t_hierarchy(r):
    """대/중/소 3단 좌측 계층 + 값. 공공·내부규정 문서의 기본형."""
    rows = ["<tr>" + cell("대분류", rs=2, hdr=True, bold=True)
            + cell("중분류", rs=2, hdr=True, bold=True)
            + cell("세부항목", rs=2, hdr=True, bold=True)
            + cell("한도", cs=2, hdr=True, bold=True)
            + cell("비고", rs=2, hdr=True, bold=True) + "</tr>",
            "<tr>" + cell("일반", hdr=True) + cell("우대", hdr=True) + "</tr>"]
    for big in r.sample(["위탁매매", "자산관리", "투자은행"], 2):
        mids = r.sample(SEGS, 2)
        for j, mid in enumerate(mids):
            subs = r.sample(["온라인", "오프라인", "모바일"], 2)
            for k, sub in enumerate(subs):
                c = ""
                if j == 0 and k == 0:
                    c += cell(big, rs=len(mids) * len(subs), hdr=True, bold=True)
                if k == 0:
                    c += cell(mid, rs=len(subs), hdr=True)
                c += cell(sub) + cell(f"{r.randint(1, 50):,}억") + cell(f"{r.randint(1, 80):,}억")
                c += cell(r.choice(["-", "", "해당없음", "별도 심사", "주1)"]))
                rows.append("<tr>" + c + "</tr>")
    return table(rows, "거래한도 기준표")


def t_kv2(r):
    """2열 구분|내용 표. 레이아웃성이 강하고 값이 긴 문장이다."""
    items = [("신청방법", "영업점 방문 또는 온라인 신청서 제출 후 본인확인 절차를 거칩니다."),
             ("처리기간", f"접수일로부터 영업일 기준 {r.randint(1, 5)}일 이내"),
             ("구비서류", "신분증, 거래인감, 법인의 경우 사업자등록증 사본 각 1부"),
             ("수수료", money(r, 10, 80)),
             ("유의사항", "미성년자 및 외국인 거주자는 별도 서류가 추가로 요구될 수 있습니다.")]
    rows = ["<tr>" + cell(k, w=140, hdr=True, bold=True) + cell(v, w=580) + "</tr>"
            for k, v in r.sample(items, r.randint(3, 5))]
    return table(rows, None)


def t_long(r):
    """60행 이상. 청크 경계가 실제로 여러 번 생긴다."""
    rows = ["<tr>" + "".join(cell(x, hdr=True, bold=True) for x in
                             ("코드", "종목명", "구분", "수수료율", "최소금액")) + "</tr>"]
    for i in range(r.randint(55, 80)):
        rows.append("<tr>" + "".join(cell(x) for x in (
            f"A{i:04d}", f"{r.choice(CORPS)}{r.choice(['1호', '2호', '플러스'])}",
            r.choice(SEGS), f"{r.uniform(0.01, 2.5):.3f}%",
            f"{r.randint(1, 500) * 10000:,}원")) + "</tr>")
    return table(rows, "상품별 수수료율")


def t_nested(r):
    """중첩 표 + 레이아웃 표. OCR/Word 양쪽에서 나온다."""
    inner = table(["<tr>" + cell("주간") + cell(f"{r.randint(9, 18)}:00") + "</tr>",
                   "<tr>" + cell("야간") + cell(f"{r.randint(18, 23)}:00") + "</tr>"], None, 200)
    rows = ["<tr>" + "".join(cell(x, hdr=True, bold=True) for x in ("구분", "운영시간", "문의")) + "</tr>",
            "<tr>" + cell("콜센터") + f"<td {STY_CEL}>{inner}</td>" + cell("1588-0000") + "</tr>",
            "<tr>" + cell("영업점") + cell("09:00 ~ 16:00") + cell("각 지점 대표번호") + "</tr>"]
    return table(rows, "고객센터 운영 안내")


SHAPES = [t_crosstab, t_period, t_records, t_hierarchy, t_kv2, t_long, t_nested]
# 교차표 원형이 실문서에서 흔하므로 가중치를 준다
WEIGHTS = [3, 3, 3, 2, 2, 1, 1]

NOTE = ("※ 본 안내는 {y}년 {m}월 기준이며 관련 규정 개정 시 변경될 수 있습니다. "
        "변경 사항은 홈페이지 공지사항을 통해 사전 안내드립니다. "
        "자세한 내용은 고객센터 또는 가까운 영업점으로 문의하시기 바랍니다.")


def make_doc(r, cid: int) -> str:
    corp = r.choice(CORPS)
    y, m = r.randint(2021, 2026), r.randint(1, 12)
    shapes = r.choices(SHAPES, weights=WEIGHTS, k=r.randint(1, 3))

    body = [f'<div style="position:absolute;left:112px;top:88px">'
            f'<span style="font-size:18px;font-weight:700">{corp} 업무안내</span><br>'
            f'<span style="font-size:11px;color:#666">CONTS{cid:04d}&nbsp;|&nbsp;'
            f'{y}.{m:02d} 개정</span></div>']
    if r.random() < 0.8:
        body.append(f"<p class=MsoNormal>{corp}의 {r.choice(SEGS)} 관련 업무 기준입니다. "
                    f"{y}년 {m}월부터 적용되며, 세부 내역은 아래와 같습니다.</p>")

    for i, sh in enumerate(shapes):
        if i and r.random() < 0.6:
            body.append(f"<p class=MsoNormal>{r.choice(['또한', '아울러', '한편'])} "
                        f"다음 사항도 함께 확인하시기 바랍니다.</p>")
        t = sh(r)
        # 표 맨 아래 전폭 안내문 행 (실파일에서 매우 흔함)
        if r.random() < 0.35:
            note = NOTE.format(y=y, m=m)
            t = t.replace("</tbody>", f'<tr><td colspan="9" {STY_HDR}>'
                                      f'<p class=MsoNormal><span>{note}</span></p></td></tr></tbody>')
        body.append(t)

    if r.random() < 0.5:
        body.append(f"<p class=MsoNormal>주1) 별도 심사가 필요한 항목은 "
                    f"영업점 상담 후 진행됩니다.</p>")

    html = ("<html xmlns:o=\"urn:schemas-microsoft-com:office:office\"><head>"
            "<meta http-equiv=Content-Type content=\"text/html; charset=utf-8\">"
            "<meta name=Generator content=\"Microsoft Word 15\">"
            f"<title>{corp} 업무안내 CONTS{cid:04d}</title></head><body lang=KO>"
            + "".join(body) + "</body></html>")

    # 실데이터 오염 재현: 일부 문서는 NFD / 전각이 섞여 들어온다
    if r.random() < 0.15:
        html = html.replace("수수료", unicodedata.normalize("NFD", "수수료"), 1)
    if r.random() < 0.15:
        html = html.replace("0", "０", 1)
    return html


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    r = random.Random(a.seed)
    out = pathlib.Path(a.out)
    out.mkdir(exist_ok=True)
    for p in out.glob("CONTS*.htm"):
        p.unlink()
    for i in range(1, a.n + 1):
        (out / f"CONTS{i:04d}.htm").write_text(make_doc(r, i), encoding="utf-8")
    tot = sum(p.stat().st_size for p in out.glob("*.htm"))
    print(f"{a.n}개 생성 -> {out}/  (총 {tot // 1024}KB, seed={a.seed})")
