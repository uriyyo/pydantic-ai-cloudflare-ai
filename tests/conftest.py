import pytest

# Every `httpx2_mock` is strict: a request without a route fails, and so does a route nobody called.
STRICT_HTTPX2 = pytest.mark.httpx2(assert_all_called=True, assert_all_mocked=True)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.get_closest_marker("httpx2") is None:
            item.add_marker(STRICT_HTTPX2)
