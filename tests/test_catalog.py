from sd import catalog as CAT


def test_catalog_lists_registered_pose_models_and_never_crashes_without_weights():
    assert isinstance(CAT.pose_models(only_ready=False), list) and all(c.id for c in CAT.pose_models(only_ready=False))
    ids = {c.id for c in CAT.pose_models(only_ready=False)}
    assert "yolo26n-pose" in ids and "rtmlib-body-lightweight" in ids
    assert "rtmlib-body-lightweight" in {c.id for c in CAT.pose_models()}        # rtmlib скачивает веса сам — всегда доступна
    assert CAT.vlm_models() == [] or all(c.id for c in CAT.vlm_models())         # Ollama не запущен — пустой список, не исключение
    assert set(CAT.summary()) == {"pose", "detector", "cycle", "photo", "vlm"}
