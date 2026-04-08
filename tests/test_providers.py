"""
Tests for the provider retry predicate.

Verifies which exceptions trigger retries and which propagate immediately.
No API calls needed -- pure unit tests on the predicate function.
"""

import pytest

from src.providers import _is_query_retriable


class TestRetriablePredicate:
    """_is_query_retriable decides whether a failed API call should be retried."""

    # --- Retriable errors (transient) ---

    def test_503_is_retriable(self):
        assert _is_query_retriable(Exception("503 UNAVAILABLE")) is True

    def test_429_rate_limit_is_retriable(self):
        assert _is_query_retriable(Exception("429 rate limit exceeded")) is True

    def test_timeout_string_is_retriable(self):
        assert _is_query_retriable(Exception("deadline exceeded timeout")) is True

    def test_unavailable_string_is_retriable(self):
        assert _is_query_retriable(Exception("service unavailable")) is True

    def test_timeout_error_is_retriable(self):
        assert _is_query_retriable(TimeoutError("read timed out")) is True

    def test_connection_error_is_retriable(self):
        assert _is_query_retriable(ConnectionError("connection refused")) is True

    def test_os_error_is_retriable(self):
        assert _is_query_retriable(OSError("network unreachable")) is True

    # --- Non-retriable errors (permanent) ---

    def test_400_bad_request_is_not_retriable(self):
        assert _is_query_retriable(Exception("400 bad request")) is False

    def test_403_permission_denied_is_not_retriable(self):
        assert _is_query_retriable(Exception("403 permission denied")) is False

    def test_404_not_found_is_not_retriable(self):
        assert _is_query_retriable(Exception("404 not found")) is False

    def test_invalid_request_is_not_retriable(self):
        assert _is_query_retriable(Exception("invalid_request: bad model name")) is False

    def test_unknown_error_is_not_retriable(self):
        assert _is_query_retriable(ValueError("something unexpected")) is False
