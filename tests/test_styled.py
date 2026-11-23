from vtuber import styled


def beats(terms):
    return [{"broll": t, "text": t} for t in terms]


def test_pick_modes():
    b = beats(["a", "b", "c", "d"])
    assert styled.pick(b, "stock") == set()
    assert styled.pick(b, "styled") == {0, 1, 2, 3}
    assert styled.pick(b, "mix") == {1, 3}          # hook stays stock, then alternate


def test_mix_keeps_split_line_together():
    # shots 1+2 are halves of one line (same b-roll term): both drawn or both stock
    b = beats(["hook", "x", "x", "y", "z"])
    assert styled.pick(b, "mix") == {1, 2, 4}


def test_clean_word_whitelist_and_whole_words():
    assert styled._clean_word("Dashboard tracking failing") == "Dashboard tracking"
    assert styled._clean_word("₹50k month") == "Rs 50k month"
    assert "<" not in styled._clean_word("<script>alert</script>")


def test_fallback_icon_from_keywords():
    assert styled._fallback("It costs fifty thousand rupees")["icon"] == "coin"
    assert styled._fallback("hmm")["icon"] == "spark"


def test_choose_style_stable():
    assert styled.choose_style("sketch", "x") == "sketch"
    assert styled.choose_style("auto", "same text") == styled.choose_style("auto", "same text")


def test_decide_follows_groq_not_a_quota():
    b = beats(["a", "b", "c", "d"])
    specs = [{"use": "drawn"}, {"use": "drawn"}, {"use": "stock"}, {"use": "drawn"}]
    assert styled.decide(b, specs) == {0, 1, 3}     # hook can be drawn, two in a row is fine


def test_decide_falls_back_per_line_when_groq_silent():
    b = beats(["a", "b", "c", "d"])
    specs = [{"use": "stock"}, {"use": None}, {"use": "stock"}, {"use": None}]
    assert styled.decide(b, specs) == {1, 3}        # unanswered lines use the alternation


def test_decide_keeps_split_line_together():
    b = beats(["a", "x", "x", "d"])
    specs = [{"use": "stock"}, {"use": "drawn"}, {"use": "stock"}, {"use": "stock"}]
    assert styled.decide(b, specs) == {1, 2}
