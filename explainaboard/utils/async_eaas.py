"""Asynchronous client for EaaS."""

from __future__ import annotations

from collections.abc import Callable
from threading import Thread
from typing import Any
import uuid

from eaas import Client, Config


class AsyncEaaSRequest:
    """An asynchronous request to EaaS."""

    def __init__(self, eaas_client: AsyncEaaSClient, request_id: str):
        """Constructor.

        Args:
            eaas_client: The client that was used in making the request.
            request_id: The ID of the request.
        """
        self._eaas_client: AsyncEaaSClient = eaas_client
        self._request_id: str = request_id
        self._result: dict | None = None

    def get_result(self) -> dict | None:
        """Fetch the result from a request that was made previously.

        Returns:
            A dictionary containing the result.
        """
        if self._result is None:
            self._result = self._eaas_client.wait_and_get_result(self._request_id)
        return self._result


# TODO(odashi): Use async concurrency to implement this functionaliry.
class AsyncEaaSClient(Client):
    """A wrapper class to support async requests for EaaS.

    It uses threads so there is a
    limit to the maximum number of parallel requests it can make.
    Example usage:
      1. `request_id = client.score([])` to start a new thread and make a request
      2. `client.wait_and_get_result(request_id)` to join the thread and get the result,
         this method can be called only once for each request_id
    """

    def __init__(self, config: Config):
        """Constructor.

        Args:
            config: The configuration for the EaaS server.
        """
        super().__init__(config)
        self._threads: dict[str, Thread] = {}
        self._results: dict[str, Any] = {}

    def _run_thread(self, original_fn: Callable[[], Any]) -> AsyncEaaSRequest:
        request_id = str(uuid.uuid1())

        def fn():
            self._results[request_id] = original_fn()

        self._threads[request_id] = Thread(target=fn)
        self._threads[request_id].start()
        return AsyncEaaSRequest(self, request_id)

    def async_score(
        self,
        inputs: list[dict],
        metrics: list[str | dict],
        calculate: list[str],
    ):
        """Score generated text asynchronously.

        Args:
            inputs: The texts to score in the dictionary form
              {"source": ..., "hypothesis": ..., "references": [..., ...]}
            metrics: The metrics to be used in scoring
            calculate: Whether to calculate on the "corpus", "stats", "sentence" level

        Returns:
            A thread ID corresponding to the thread doing the calculation
        """
        return self._run_thread(
            lambda: super(AsyncEaaSClient, self).score(inputs, metrics, calculate)
        )

    def wait_and_get_result(self, request_id: str) -> dict:
        """Wait for the request to complete and get the result.

        Args:
            request_id: The ID of the request

        Returns:
            A dictionary of results from the request.
        """
        if request_id not in self._threads:
            raise Exception(f"thread_id {request_id} doesn't exist")
        self._threads[request_id].join()
        result = self._results[request_id]
        self._results.pop(request_id)
        self._threads.pop(request_id)
        return result
