"""Все профили из configs/experiments загружаются: ключи конфигурации и options существуют (опечатка = ошибка), классификатор задан."""
import pytest

from sd import profiles as PR


@pytest.mark.parametrize("name", PR.list_profiles())
def test_shipped_profile_loads(name):
    p = PR.load(name)
    p.cfg()
    p.opts()
    assert p.classifier_problem() is None, name
