import os
import hashlib
import json
from pathlib import Path
import math
from collections import deque

from typing import Iterator, List, Optional, Deque, TypeVar, Protocol, Union
from abc import ABC, abstractmethod

from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


T = TypeVar("T")


class BatchedIterator(Protocol[T], Iterator[T]):

    def set_batch_size(self, n: int) -> None:
        raise NotImplementedError()


class Model(ABC):

    def __init__(self,
                 model_name: str,
                 temperature: float,
                 alias: Optional[str] = None,
                 max_batch: int = 1):
        self.model_name = model_name
        self.temperature = temperature
        if alias is not None:
            self.alias = alias
        else:
            self.alias = model_name
        self.max_batch = max_batch

    @abstractmethod
    def sample(self, prompt: str, batch: int = 1) -> BatchedIterator[str]:
        raise NotImplementedError()


class ReplicationCacheMiss(Exception):
    "Raised when a cache miss occurs in replication mode"
    pass


class _BaseBufferedModel(Model):
    """A base model with buffered queries abstracted via _query method"""

    def __init__(self,
                 model_name: str,
                 temperature: float,
                 alias: Optional[str] = None,
                 max_batch: int = 1):
        super().__init__(model_name, temperature, alias, max_batch)

    @abstractmethod
    def _query(self, prompt: str, n: int) -> List[str]:
        raise NotImplementedError()

    class _BufferedIterator(BatchedIterator):

        def __init__(self, base, prompt):
            self.base = base
            self.prompt = prompt
            self._buffer: Deque[str] = deque()

        def __iter__(self):
            return self

        def set_batch_size(self, n: int) -> None:
            batches = max(1, math.ceil(n / self.base.max_batch))
            self.batch_size = math.ceil(n / batches)            

        def __next__(self):
            if len(self._buffer) == 0:
                responses = self.base._query(self.prompt, self.batch_size)
                self._buffer.extend(responses)            
            return self._buffer.popleft()        

    def sample(self, prompt: str, batch: int = 1) -> BatchedIterator[str]:
        i = _BaseBufferedModel._BufferedIterator(self, prompt)
        i.set_batch_size(batch)
        return i


class OpenAICompatibleHTTPModel(_BaseBufferedModel):
    """
    Expected endpoint: POST {base_url}/chat/completions
    Headers:
      - Authorization: Bearer <api_key>
      - Content-Type: application/json

    Request body:
      {
        "model": "<model_id>",
        "temperature": <float>,
        "n": <int>,
        "messages": [{"role": "user", "content": "<prompt>"}]
      }

    Response body (non-streaming):
      {
        "choices": [
          { "index": 0, "message": { "role": "assistant", "content": "..." } },
          ...
        ]
      }
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float,
        alias: Optional[str] = None,
        max_batch: int = 1
    ):
        super().__init__(model_name, temperature, alias, max_batch)
        self.base_url = base_url
        self.api_key = api_key

    def _post_json(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            'Accept': 'application/json',            
            "User-Agent": "Application"
        }
        req = Request(url, data=data, headers=headers, method="POST")
        try:
            with urlopen(req) as resp:
                raw = resp.read()
                return json.loads(raw.decode("utf-8"))
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTPError {e.code} {e.reason}: {body}") from e
        except URLError as e:
            raise RuntimeError(f"URLError: {e.reason}") from e

    def _query(self, prompt: str, n: int) -> List[str]:
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "n": n,
            "messages": [{"role": "user", "content": prompt}]
        }
        resp = self._post_json("/chat/completions", payload)
        return [str(c["message"]["content"]) for c in resp["choices"]]


class FireworksAI(OpenAICompatibleHTTPModel):
    def __init__(self, model_name: str, temperature: float, alias: Optional[str] = None, max_batch: int = 1):
        base_url = "https://api.fireworks.ai/inference/v1"
        api_key = os.environ["FIREWORKS_API_KEY"]
        super().__init__(base_url, api_key, model_name, temperature, alias, max_batch)


class AI302(OpenAICompatibleHTTPModel):
    def __init__(self, model_name: str, temperature: float, alias: Optional[str] = None, max_batch: int = 1):
        base_url = "https://api.302.ai/v1"
        api_key = os.environ["AI302_API_KEY"]
        super().__init__(base_url, api_key, model_name, temperature, alias, max_batch)


class CloseAI(OpenAICompatibleHTTPModel):
    def __init__(self, model_name: str, temperature: float, alias: Optional[str] = None, max_batch: int = 1):
        base_url = "https://api.openai-proxy.org/v1"
        api_key = os.environ["CLOSEAI_API_KEY"]
        super().__init__(base_url, api_key, model_name, temperature, alias, max_batch)


class XMCP(OpenAICompatibleHTTPModel):
    def __init__(self, model_name: str, temperature: float, alias: Optional[str] = None, max_batch: int = 1):
        base_url = "https://llm.xmcp.ltd"
        api_key = os.environ["XMCP_API_KEY"]
        super().__init__(base_url, api_key, model_name, temperature, alias, max_batch)


class Independent(Model):
    """Ensure that each call of sample returns an independent sequence"""

    def __init__(self, inner: Model):
        super().__init__(inner.model_name,
                         inner.temperature,
                         inner.alias,
                         inner.max_batch)
        self._inner = inner
        self._inner_iters = {}  # prompt_hash -> sample sequence

    def sample(self, prompt: str, batch: int = 1) -> BatchedIterator[str]:
        if isinstance(self._inner, Independent) or \
           isinstance(self._inner, OpenAICompatibleHTTPModel) or \
           isinstance(self._inner, XMCP) or \
           isinstance(self._inner, CloseAI) or \
           isinstance(self._inner, FireworksAI) or \
           isinstance(self._inner, AI302):
            return self._inner.sample(prompt, batch)
        # for the same prompt, always return the same iterator
        pid = prompt_id(prompt)
        if pid not in self._inner_iters:
            self._inner_iters[pid] = self._inner.sample(prompt, batch)
        self._inner_iters[pid].set_batch_size(batch)
        return self._inner_iters[pid]


class _BaseBatchedCache(Model):
    """A cache that supports batch sampling, while abstracting the storage"""

    @abstractmethod
    def _store(self, pid: str, response: str):
        raise NotImplementedError()

    @abstractmethod
    def _load(self, pid: str) -> list[str]:
        raise NotImplementedError()
    
    def __init__(self, inner: Model, fail_on_miss: bool = False):
        super().__init__(inner.model_name,
                         inner.temperature,
                         inner.alias,
                         inner.max_batch)
        # the cache calls the inner model only when it needs a fresh sample:
        self._inner = Independent(inner)
        self.fail_on_miss = fail_on_miss

    class _SharedCacheIterator(BatchedIterator[str]):
        def __init__(self, base, prompt: str):
            self.base = base
            self.prompt = prompt
            self.pid = prompt_id(prompt)
            self.batch_size = 1
            self.current_index = 0

        def set_batch_size(self, n: int) -> None:
            self.batch_size = n

        def __iter__(self):
            return self

        def __next__(self) -> str:
            cache = self.base._load(self.pid)
            if len(cache) > self.current_index:
                self.current_index += 1
                return cache[self.current_index - 1]
            if self.base.fail_on_miss:
                raise ReplicationCacheMiss()
            fresh = next(self.base._inner.sample(self.prompt, self.batch_size))
            self.base._store(self.pid, fresh)
            self.current_index += 1
            return fresh

    def sample(self, prompt: str, batch: int = 1) -> BatchedIterator[str]:
        i = _BaseBatchedCache._SharedCacheIterator(self, prompt)
        i.set_batch_size(batch)
        return i


class Repeatable(_BaseBatchedCache):
    """An in-memory cache a.k.a. return the same sequence for each call of sample"""

    def __init__(self, inner: Model):
        super().__init__(inner)
        self._cache = dict() # prompt_id -> list of responses

    def _store(self, pid: str, response: str):
        if pid not in self._cache:
            self._cache[pid] = []
        self._cache[pid].append(response)

    def _load(self, pid: str) -> list[str]:
        return self._cache.get(pid, [])

    def sample(self, prompt: str, batch: int = 1) -> BatchedIterator[str]:
        if isinstance(self._inner, Repeatable) or isinstance(self._inner, Persistent):
            return self._inner.sample(prompt, batch)
        return super().sample(prompt, batch)
    

class Persistent(_BaseBatchedCache):
    """An on-disk cache that saves responses in the following filesystem format:

    <cache_root>/
      <model_alias>_<temperature>/
        <prompt_hash>/
          0.md
          1.md
          ...
    """
    def __init__(self, inner: Model, cache_root: Union[Path, str], replication: bool = False):
        super().__init__(inner, replication)
        if isinstance(cache_root, str):
            cache_root = Path(cache_root)
        self.cache_root = cache_root
        self.replication = replication

    def _store(self, pid: str, response: str):
        d = self._prompt_dir(pid)
        d.mkdir(parents=True, exist_ok=True)
        i = len(Persistent._list_numbered_files(d))
        f = d / f"{i}.md"
        f.write_text(response)
 
    def _load(self, pid: str) -> list[str]:
        d = self._prompt_dir(pid)
        return [f.read_text() for f in Persistent._list_numbered_files(d)]

    def _prompt_dir(self, pid: str) -> str:
        t = f"{self.temperature:.3f}".rstrip("0").rstrip(".")
        model_key = f"{self.alias}_{t}"
        return self.cache_root / model_key / pid

    @staticmethod
    def _list_numbered_files(path: Path) -> List[str]:
        if not path.is_dir():
            return []
        md_files = list(path.glob('*.md'))
        return sorted(md_files, key=lambda f: int(f.stem))
