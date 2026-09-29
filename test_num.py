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


def test_RRF_융합():
    """순위 융합. 두 검색기가 엇갈릴 때 '둘 다 웬만큼 좋은' 문서가 이겨야 한다."""
    import numpy as np
    from evaluate import HybridRRF, Retriever

    class Fake(Retriever):
        def __init__(self, order): self.order = order
        def ranks(self, queries, docs): return np.array([self.order] * len(queries))

    docs = [str(i) for i in range(10)]
    #  0번: A 1등 / B 꼴찌,  9번: B 1등 / A 꼴찌   (한쪽만 확신)
    #  1번: 양쪽 모두 2등                          (둘 다 웬만큼)
    # -> 한쪽 확신보다 양쪽 합의가 이겨야 한다
    a = Fake([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = Fake([9, 1, 8, 7, 6, 5, 4, 3, 2, 0])
    fused = HybridRRF(a, b).ranks(["q"], docs)[0]
    assert docs[fused[0]] == "1", [docs[i] for i in fused]
    # 한쪽 1등짜리 둘은 뒤로 밀린다
    assert fused.tolist().index(0) > 0 and fused.tolist().index(9) > 0

    # 두 검색기가 같으면 융합해도 순서가 그대로다
    same = HybridRRF(a, Fake(list(range(10)))).ranks(["q"], docs)[0]
    assert list(same) == list(range(10)), same

    # 모든 문서가 정확히 한 번씩 나온다 (유실·중복 금지)
    assert sorted(fused) == list(range(len(docs)))

    # RRF 를 쓰는 이유: 순위가 포화한다. 한쪽에서 500등이어도 다른 쪽 1등이면
    # 살아남는다. 단순 순위합(Borda)이면 '500' 이 그대로 더해져 뭉개버린다.
    n, X, Y = 600, 0, 1
    rest = list(range(2, n))
    oa = [X] + rest[:49] + [Y] + rest[49:]            # X 0등, Y 50등
    ob = rest[:50] + [Y] + rest[50:498] + [X] + rest[498:]   # Y 50등, X 499등
    assert oa.index(X) == 0 and oa.index(Y) == 50
    assert ob.index(Y) == 50 and ob.index(X) == 499
    big = HybridRRF(Fake(oa), Fake(ob)).ranks(["q"], [str(i) for i in range(n)])[0]
    assert big.tolist().index(X) < big.tolist().index(Y), "RRF 가 포화하지 않는다"


def test_검색기_조합_구성():
    """CLI 플래그 -> 검색기 조합. --hybrid 를 밀집 없이 쓰면 막아야 한다."""
    import argparse
    from evaluate import CharTfidf, HybridRRF, build_retriever, describe

    mk = lambda **kw: argparse.Namespace(
        **{"st": None, "hybrid": False, "rerank": None, "rerank_top": 30, **kw})

    assert isinstance(build_retriever(mk()), CharTfidf)
    assert describe(build_retriever(mk())) == "CharTfidf"
    try:
        build_retriever(mk(hybrid=True))          # 밀집 없이 융합 불가
    except SystemExit:
        pass
    else:
        raise AssertionError("--hybrid 를 --st 없이 허용했다")


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
        good = L.run([html], "_gold", "", limit=4, verbose=False)
        bad = L.run([html], "_wrong", "", limit=4, verbose=False)
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
