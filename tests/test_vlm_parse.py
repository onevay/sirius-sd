import math

from sd.vlm import load_prompt, parse_answer, smoke_score


def test_parse_clean_json():
    a = parse_answer('{"action": "smoking", "object_visible": true, "smoke_visible": false}')
    assert a["parsed"] and a["action"] == "smoking" and a["object_visible"] is True and a["smoke_visible"] is False
    assert smoke_score(a) == 1.0


def test_parse_json_inside_text_and_vaping():
    a = parse_answer('Here is the answer:\n```json\n{"action": "vaping", "object_visible": false, "smoke_visible": true}\n```')
    assert a["action"] == "vaping" and smoke_score(a) == 1.0


def test_unknown_action_becomes_unclear_and_nan():
    a = parse_answer('{"action": "dancing", "object_visible": false, "smoke_visible": false}')
    assert a["action"] == "unclear" and math.isnan(smoke_score(a))


def test_negative_actions_score_zero():
    for act in ("drinking", "phone", "eating", "touching_face", "other"):
        assert smoke_score(parse_answer('{"action": "%s"}' % act)) == 0.0


def test_garbage_falls_back_to_keywords_then_unclear():
    assert parse_answer("the person is drinking from a bottle")["action"] == "drinking"
    assert parse_answer("???")["action"] == "unclear"


def test_prompt_template_filled():
    p = load_prompt(6, 2.5)
    assert "6 frames" in p and "2.5 s" in p and "JSON" in p
