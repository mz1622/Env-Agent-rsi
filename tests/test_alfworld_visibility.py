"""The optional panoramic data must never expose closed-container contents."""
from types import SimpleNamespace

from envharness.bridges.alfworld.bridge import AlfworldEnv


def _fact(name: str, *arguments: str):
    return SimpleNamespace(
        name=name,
        arguments=[SimpleNamespace(name=arg) for arg in arguments],
    )


def test_open_surface_contents_exclude_closed_and_stale_facts():
    facts = [
        _fact("receptacleatlocation", "table 1", "loc 1"),
        _fact("receptacleatlocation", "cabinet 1", "loc 2"),
        _fact("openable", "cabinet 1"),
        _fact("objectatlocation", "kettle 1", "loc 1"),
        _fact("objectatlocation", "mug 1", "loc 2"),
        _fact("inreceptacle", "kettle 1", "table 1"),
        _fact("inreceptacle", "kettle 1", "cabinet 1"),
        _fact("inreceptacle", "mug 1", "cabinet 1"),
        _fact("inreceptacle", "mug 1", "table 1"),
    ]
    assert AlfworldEnv._surface_contents_from_facts(facts) == {
        "table 1": ["kettle 1"],
    }
