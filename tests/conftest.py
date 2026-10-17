"""Share the framework's `env` fixture (tests/conftest.py of the outer repo) with
these tests: they live in the `research/` checkout, outside that conftest's scope."""

from tests.conftest import env  # noqa: F401
