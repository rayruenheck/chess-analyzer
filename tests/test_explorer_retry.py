"""Rate-limit handling for the Lichess Opening Explorer.

A single 429 partway through a report used to kill the opening analysis for every
remaining game, because the caller gives up on the whole feature rather than
retrying. These pin the backoff that keeps a slow answer instead of no answer.
"""
import asyncio, httpx, pytest
from app.clients.lichess_explorer import LichessExplorerClient

def make(responses):
    calls = {"n": 0}
    def handler(request):
        calls["n"] += 1
        return responses[min(calls["n"]-1, len(responses)-1)]
    c = LichessExplorerClient()
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                                  base_url="https://explorer.lichess.org")
    c.DEFAULT_RETRY_AFTER = 0.01
    c.MIN_REQUEST_INTERVAL = 0.0
    c._interval = 0.0
    return c, calls

def test_a_429_is_retried_not_surfaced():
    ok = httpx.Response(200, json={"white": 1, "draws": 0, "black": 0, "moves": []})
    c, calls = make([httpx.Response(429), httpx.Response(429), ok])
    out = asyncio.run(c.get_lichess("x", ratings=[1400], speeds=["rapid"]))
    assert out["white"] == 1
    assert calls["n"] == 3, "should have retried twice before succeeding"

def test_rate_limiting_slows_the_client_for_the_rest_of_the_session():
    """A report walks hundreds of positions; rediscovering the limit each time
    would 429 on nearly all of them."""
    ok = httpx.Response(200, json={"moves": []})
    c, _ = make([httpx.Response(429), ok])
    before = c._interval = 0.5
    asyncio.run(c.get_lichess("x"))
    assert c._interval > before

def test_persistent_rate_limiting_still_gives_up():
    c, calls = make([httpx.Response(429)])
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(c.get_lichess("x"))
    assert calls["n"] == c.MAX_RETRIES + 1

def test_retry_after_header_is_honoured():
    c, _ = make([httpx.Response(429, headers={"Retry-After": "2.5"})])
    assert c._retry_after(httpx.Response(429, headers={"Retry-After": "2.5"})) == 2.5
    assert c._retry_after(httpx.Response(429)) is None
