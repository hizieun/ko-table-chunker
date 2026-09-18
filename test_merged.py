"""병합 셀(rowspan/colspan) 전개 테스트.

그리드 좌표 단위로 검증한다. 청크 문자열만 보면 점유 맵의 off-by-one 이
"대충 맞아 보이는" 결과로 숨는다.

실행: python3 test_merged.py
"""

from bs4 import BeautifulSoup

from html_chunker import BACKEND, build_table, parse, row_kv

BLANK = (-1, -1)


def T(html: str):
    return build_table(BeautifulSoup(html, BACKEND).find("table"))


def grid_text(t):
    return [[c.text for c in r] for r in t.grid]


# --------------------------------------------------------------------------- #

def test_rowspan_전방채움():
    """rowspan 은 아래 행 전부에 값을 복제해야 한다. 이게 안 되면
    2·3번째 행은 '반도체' 라는 문자열이 없어 영원히 검색되지 않는다."""
    t = T("""<table>
      <tr><th>부문</th><th>매출</th></tr>
      <tr><td rowspan="3">반도체</td><td>100</td></tr>
      <tr><td>200</td></tr>
      <tr><td>300</td></tr>
    </table>""")
    assert grid_text(t) == [["부문", "매출"], ["반도체", "100"],
                           ["반도체", "200"], ["반도체", "300"]], grid_text(t)
    # 복제된 셀은 원본과 같은 origin 을 공유한다
    assert t.grid[1][0].origin == t.grid[2][0].origin == t.grid[3][0].origin == (1, 0)
    # 각 행이 독립적으로 검색 가능해야 한다
    assert row_kv(t.labels, t.body[1]) == "부문: 반도체, 매출: 200"


def test_colspan_중복제거():
    """colspan 으로 복제된 셀은 KV 에 한 번만 나와야 한다."""
    t = T("""<table>
      <tr><th>구분</th><th>상반기</th><th>하반기</th></tr>
      <tr><td>비고</td><td colspan="2">해당없음</td></tr>
    </table>""")
    assert grid_text(t)[1] == ["비고", "해당없음", "해당없음"]
    assert t.grid[1][1].origin == t.grid[1][2].origin == (1, 1)
    kv = row_kv(t.labels, t.body[0])
    assert kv.count("해당없음") == 1, kv
    assert kv == "구분: 비고, 상반기: 해당없음", kv


def test_다단헤더_합성():
    """상위 헤더(colspan)와 하위 헤더를 이어 컬럼 라벨을 만든다."""
    t = T("""<table>
      <tr><th rowspan="2">부문</th><th colspan="2">2024년</th><th rowspan="2">증감</th></tr>
      <tr><th>상반기</th><th>하반기</th></tr>
      <tr><td>반도체</td><td>100</td><td>200</td><td>5%</td></tr>
    </table>""")
    assert t.n_header == 2, t.n_header
    assert t.labels == ["부문", "2024년 > 상반기", "2024년 > 하반기", "증감"], t.labels
    # 헤더의 rowspan 은 두 행 모두 채우되 라벨은 중복 제거 ("부문 > 부문" 금지)
    assert t.grid[0][0].text == t.grid[1][0].text == "부문"
    assert t.labels[0] == "부문"


def test_2x2_블록병합():
    """rowspan 과 colspan 이 동시에 걸린 셀은 4칸을 모두 채운다."""
    t = T("""<table>
      <tr><th>A</th><th>B</th><th>C</th></tr>
      <tr><td rowspan="2" colspan="2">덩어리</td><td>x1</td></tr>
      <tr><td>x2</td></tr>
    </table>""")
    assert grid_text(t) == [["A", "B", "C"], ["덩어리", "덩어리", "x1"],
                           ["덩어리", "덩어리", "x2"]], grid_text(t)
    origins = {t.grid[r][c].origin for r in (1, 2) for c in (0, 1)}
    assert origins == {(1, 0)}, origins
    assert row_kv(t.labels, t.body[0]) == "A: 덩어리, C: x1"


def test_계단식_계층():
    """좌측 3단 계층(대/중/소분류). 한국 공공문서의 기본형.
    rowspan 이 서로 어긋나게 겹치는 배치에서 열이 밀리면 안 된다."""
    t = T("""<table>
      <tr><th>대분류</th><th>중분류</th><th>세부</th><th>예산</th></tr>
      <tr><td rowspan="4">첨단소재</td><td rowspan="2">반도체</td><td>메모리</td><td>3,890</td></tr>
      <tr><td>화합물</td><td>2,140</td></tr>
      <tr><td rowspan="2">이차전지</td><td>전고체</td><td>2,780</td></tr>
      <tr><td>리튬메탈</td><td>1,120</td></tr>
      <tr><td rowspan="2">바이오</td><td rowspan="2">첨단바이오</td><td>유전자</td><td>2,010</td></tr>
      <tr><td>파운드리</td><td>1,340</td></tr>
    </table>""")
    assert [len(r) for r in t.grid] == [4] * 7, [len(r) for r in t.grid]
    assert grid_text(t)[1:] == [
        ["첨단소재", "반도체", "메모리", "3,890"],
        ["첨단소재", "반도체", "화합물", "2,140"],
        ["첨단소재", "이차전지", "전고체", "2,780"],
        ["첨단소재", "이차전지", "리튬메탈", "1,120"],
        ["바이오", "첨단바이오", "유전자", "2,010"],
        ["바이오", "첨단바이오", "파운드리", "1,340"],
    ], grid_text(t)[1:]
    # 마지막 행만 봐도 계층 전체가 복원된다 = 이 행 하나로 검색이 된다
    assert row_kv(t.labels, t.body[-1]) == \
        "대분류: 바이오, 중분류: 첨단바이오, 세부: 파운드리, 예산: 1,340"


def test_소계행_colspan라벨():
    """'소계' 가 앞 두 컬럼을 덮는 배치. 값이 엉뚱한 컬럼으로 밀리기 쉽다."""
    t = T("""<table>
      <tr><th>부문</th><th>제품</th><th>매출</th><th>이익</th></tr>
      <tr><td rowspan="2">반도체</td><td>메모리</td><td>100</td><td>10</td></tr>
      <tr><td>파운드리</td><td>200</td><td>20</td></tr>
      <tr><td colspan="2">소계</td><td>300</td><td>30</td></tr>
    </table>""")
    assert grid_text(t)[-1] == ["소계", "소계", "300", "30"], grid_text(t)[-1]
    kv = row_kv(t.labels, t.body[-1])
    assert kv == "부문: 소계, 매출: 300, 이익: 30", kv   # 300 이 '제품' 으로 밀리면 실패


def test_병합이_표폭을_넘음():
    """colspan 이 실제 열 수보다 큰 경우(OCR 흔함). 죽지 않고 그리드는 직사각."""
    t = T("""<table>
      <tr><td>A</td><td>B</td><td>C</td></tr>
      <tr><td colspan="9">넘침</td></tr>
      <tr><td>x</td><td>y</td><td>z</td></tr>
    </table>""")
    assert len({len(r) for r in t.grid}) == 1, [len(r) for r in t.grid]
    assert len(t.grid[0]) == 9
    assert all(c.text == "넘침" for c in t.grid[1]), grid_text(t)[1]
    # 짧은 행은 빈 칸으로 메워지고, 그 칸은 원본 셀이 아님을 origin 으로 구분한다
    assert t.grid[0][3].origin == BLANK and t.grid[0][0].origin != BLANK


def test_이상한_span_값():
    """rowspan="0" / 숫자 아님 / 음수. 스펙보다 안전한 클램프를 택한다."""
    t = T("""<table>
      <tr><td>A</td><td>B</td></tr>
      <tr><td rowspan="0">영</td><td>1</td></tr>
      <tr><td rowspan="abc">문자</td><td>2</td></tr>
      <tr><td colspan="-3">음수</td><td>3</td></tr>
    </table>""")
    assert [len(r) for r in t.grid] == [2] * 4, [len(r) for r in t.grid]
    assert grid_text(t)[1:] == [["영", "1"], ["문자", "2"], ["음수", "3"]], grid_text(t)


def test_병합셀이_청크경계를_넘어감():
    """표가 여러 청크로 쪼개져도 각 청크의 모든 행이 병합 값을 갖는다.
    이게 깨지면 뒤쪽 청크의 행들은 부문명 없이 숫자만 남는다."""
    rows = "".join(f"<tr><td>{i*100}</td><td>{i*7}%</td></tr>" for i in range(1, 40))
    html = f"""<table><caption>긴 표</caption>
      <tr><th>부문</th><th>매출</th><th>비중</th></tr>
      <tr><td rowspan="40">반도체</td><td>0</td><td>0%</td></tr>
      {rows}</table>"""
    chunks = [c for c in parse(html, max_chars=400) if c.kind == "table"]
    assert len(chunks) >= 3, f"청크가 {len(chunks)}개뿐 — 경계 테스트가 안 됨"
    for c in chunks:
        for line in c.text.splitlines():
            if line.startswith("부문:"):
                assert "부문: 반도체" in line, line
    # 마지막 청크까지 부문명이 살아있는가
    assert "부문: 반도체" in chunks[-1].text


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"\n{len(tests)} passed")
