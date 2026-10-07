"""Выбор клипов HMDB51 из списка файлов зеркала (без сети): класс по префиксу имени, детерминированная выборка."""
from sd import video_fetch as VF

FILES = ([f"train/smoke_clip{i}_smoke_u_nm_np1_fr_goo_{i}.mp4" for i in range(10)] + [f"test/smoke_x{i}_smoke_f_cm_np1_le_med_{i}.mp4" for i in range(4)]
         + [f"train/drink_a{i}_drink_u_nm_np1_ba_goo_0.mp4" for i in range(6)] + [f"validation/brush_hair_b{i}_brush_hair_u_nm_np1_ri_bad_1.mp4" for i in range(3)]
         + ["train/metadata.csv", "README.md", "train/smoke_note.txt", "train/smokestack_x_smokestack_u.mp4"])


def test_class_of_matches_prefix_with_underscore_and_prefers_longest():
    assert VF.class_of("train/smoke_a_smoke_u_nm.mp4", ["smoke", "drink"]) == "smoke"
    assert VF.class_of("validation/brush_hair_b_brush_hair_u.mp4", ["brush", "brush_hair"]) == "brush_hair"     # длинный класс раньше короткого
    assert VF.class_of("train/smokestack_x.mp4", ["smoke"]) is None                                            # префикс без «_» — другой класс/не наш
    assert VF.class_of("train/eat_a_eat_u.mp4", ["smoke"]) is None


def test_select_filters_non_video_and_limits_per_class_deterministically():
    a = VF.select(FILES, ["smoke", "drink", "eat"], per_class=5, seed=0)
    b = VF.select(FILES, ["smoke", "drink", "eat"], per_class=5, seed=0)
    assert a == b and len(a["smoke"]) == 5 and len(a["drink"]) == 5 and a["eat"] == []                          # drink: 6 → 5; клипов eat в списке нет
    assert all(f.endswith(".mp4") and "smoke_" in f.split("/")[-1][:6] for f in a["smoke"])
    assert VF.select(FILES, ["smoke"], per_class=None, seed=0)["smoke"] == sorted(f for f in FILES if f.split("/")[-1].startswith("smoke_") and f.endswith(".mp4"))
    assert len(VF.select(FILES, ["brush_hair"])["brush_hair"]) == 3
