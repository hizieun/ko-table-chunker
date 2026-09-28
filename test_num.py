"""한국어 수량 정규화 테스트.

L4 채점이 이 함수 하나에 걸려 있다. 여기가 틀리면 '정답률 0.9' 가 통째로 거짓이 된다.
실행: python3 test_num.py
"""

from l4_extract import normalize_num as N, same_value as S


def test_기본형():
    assert N("55000") == 55000
    assert N("55,000") == 55000
    assert N("연간 55,000원") == 55000
    assert N("2,840,000") == 2840000
    assert N("3.5") == 3.5


def test_부호():
    assert N("-8.1") == -8.1
    assert N("△380") == -380          # 회계 문서의 음수 표기
    assert N("▲1,200") == -1200
    assert N("-8.1%") == -8.1


def test_전각_유니코드():
    assert N("４,３２０") == 4320      # NFKC 로 전각 접힘
    assert N("５．１％") == 5.1


def test_한글_큰단위():
    assert N("3.5억") == 350_000_000
    assert N("12조") == 12_000_000_000_000
    assert N("5만") == 50_000


def test_숫자없음():
    for s in ["-", "해당없음", "전 제휴 서비스/기관", "", None]:
        assert N(s) is None, s


def test_동일값_판정():
    assert S("연간 55,000원", "55000")
    assert S("55,000", "연간 55,000 원")
    assert S("100.0%", "100.0")
    assert not S("연간 55,000원", "연간 3,300원")
    assert not S("3,300", "33,000")
    # 비숫자는 문자열 비교 (공백·유니코드 정규화 후)
    assert S("전 제휴 서비스/기관", "전제휴서비스/기관")
    assert not S("전 계열사", "전 은행")
    assert not S("", "")              # 빈 응답은 정답이 될 수 없다
    assert not S("NONE", "55000")


def test_사용자_실패케이스():
    """55,000 과 3,300 을 절대 같다고 하면 안 된다 — 이 프로젝트의 존재 이유."""
    assert not S("연간 3,300원", "연간 55,000원")
    assert S("연간 55,000원", "연간 55,000원")


def test_복합_한글수사():
    """자리올림이 중첩되는 형태. (숫자,단위) 쌍을 그냥 더하면 천만 배 틀린다."""
    assert N("2천만") == 20_000_000          # 순진한 구현은 2.0 을 낸다
    assert N("1억 2천만") == 120_000_000
    assert N("3천5백") == 3500
    assert N("백만") == 1_000_000            # 앞에 숫자가 없으면 1
    assert N("1조 2천억") == 1_200_000_000_000
    assert N("1천2백") == 1200


def test_L4_채점_배선():
    """LLM 없이 L4 경로(검색 -> 프롬프트 -> 채점)가 맞는지 고정한다.

    정답을 그대로 뱉는 백엔드는 1.000, 틀린 값을 뱉는 백엔드는 0.000 이어야 한다.
    이게 안 맞으면 실제 LLM 점수도 전부 거짓이다.
    """
    import l4_extract as L
    from evaluate import gen_queries

    html = open("fixtures/05_워드내보내기.html", encoding="utf-8").read()
    qs = gen_queries(html)[:4]
    answers = {q.q: q.answer for q in qs}

    L.BACKENDS["_gold"] = lambda p, m, **k: answers[p.split("[질문] ")[1].split("\n")[0]]
    L.BACKENDS["_wrong"] = lambda p, m, **k: "연간 3,300원"
    try:
        good = L.run([html], "_gold", "", limit=4)
        bad = L.run([html], "_wrong", "", limit=4)
    finally:
        L.BACKENDS.pop("_gold", None)
        L.BACKENDS.pop("_wrong", None)

    assert all(v == 1.0 for v in good.values()), good
    assert all(v < 1.0 for v in bad.values()), bad


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"\n{len(tests)} passed")
