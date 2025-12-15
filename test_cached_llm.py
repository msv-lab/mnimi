from itertools import islice
from cached_llm import (
    prompt_id,
    Repeatable,
    Independent,
    Persistent,
    _BaseBufferedModel,
    Model,
    BatchedIterator
)


class MockModel(Model):
    def __init__(self, responses):
        self.responses = responses
        self.num_iterated = 0
        super().__init__("mock", 1.0)

    class _MockIterator(BatchedIterator):

        def __init__(self, base, prompt):
            self.base = base
            self.prompt = prompt
            self.index = 0

        def __iter__(self):
            return self

        def set_batch_size(self, n: int) -> None:
            self.batch_size = n

        def __next__(self):
            self.index += 1
            self.base.num_iterated += 1
            return self.base.responses[self.prompt][self.index - 1]

    def sample(self, prompt: str, batch: int = 1) -> BatchedIterator[str]:
        i = MockModel._MockIterator(self, prompt)
        i.set_batch_size(batch)
        return i

    def total_query_time(self):
        pass

    def total_token_count(self):
        pass


def test_persistent(tmp_path):
    m = MockModel({ "prompt": [ "0", "1", "2", "3", "4" ] })
    c = Persistent(m, tmp_path)
    s1 = c.sample("prompt")
    s2 = c.sample("prompt")
    assert next(s1) == "0"
    assert next(s2) == "0"
    assert next(s1) == "1"
    assert next(s2) == "1"
    assert next(s1) == "2"
    assert m.num_iterated == 3


def test_repeatable():
    m = MockModel({ "prompt": [ "0", "1", "2", "3", "4" ] })
    c = Repeatable(m)
    s1 = c.sample("prompt")
    s2 = c.sample("prompt")
    assert next(s1) == "0"
    assert next(s2) == "0"
    assert next(s1) == "1"
    assert next(s2) == "1"
    assert next(s1) == "2"
    assert m.num_iterated == 3


def test_repeatable_is_stateless():
    m = MockModel({ "prompt": [ "0", "1", "2", "3", "4" ] })
    c = Repeatable(m)
    s1 = c.sample("prompt")
    assert next(s1) == "0"
    assert next(s1) == "1"
    s2 = c.sample("prompt")
    assert next(s2) == "0"


def test_independent():
    m = Repeatable(MockModel({ "prompt": [ "0", "1", "2", "3", "4" ] }))
    ind = Independent(m)
    responses = []
    for i in range(2):
        r = Repeatable(ind)
        s1 = r.sample("prompt")
        s2 = r.sample("prompt")
        responses.append(next(s1))
        responses.append(next(s2))
        responses.append(next(s1))
    assert responses == ["0", "0", "1", "2", "2", "3"]


def test_nested_cache(tmp_path):
    m1 = MockModel({ "prompt": [ "0", "1" ] })
    c1 = Persistent(m1, f"{tmp_path}/a")
    s1 = c1.sample("prompt")
    next(s1)
    next(s1)
    assert m1.num_iterated == 2

    m2 = MockModel({ "prompt": [ "0", "1" ] })
    c2 = Persistent(m2, f"{tmp_path}/a")
    c2_nested = Persistent(c2, f"{tmp_path}/b")
    s2 = c2_nested.sample("prompt")
    next(s2)
    assert m2.num_iterated == 0

    m3 = MockModel({ "prompt": [ "0", "1" ] })
    c3 = Persistent(m3, f"{tmp_path}/b")
    s3 = c3.sample("prompt")
    next(s3)
    next(s3)
    assert m3.num_iterated == 1



class MockBufferedModel(_BaseBufferedModel):

    def __init__(self, responses, max_batch):
        super().__init__("mock", 1.0, max_batch=max_batch)
        self.responses = responses
        self.current_indexes = dict()
        for prompt in responses:
            self.current_indexes[prompt] = 0
        self.num_queries = 0

    def _query(self, prompt: str, n: int):
        self.num_queries += 1
        index = self.current_indexes[prompt]
        responses = self.responses[prompt][index:index + n]
        self.current_indexes[prompt] = index + n
        return responses

    def total_query_time(self):
        pass

    def total_token_count(self):
        pass


def test_batched():
    m = MockBufferedModel({ "prompt": [ "0", "1", "2", "3", "4" ] }, max_batch=2)
    responses = []
    for r in islice(m.sample("prompt", batch=2), 4):
        responses.append(r)
    assert responses == ["0", "1", "2", "3"]
    assert m.num_queries == 2


def test_batched_limit():
    m = MockBufferedModel({ "prompt": [ "0", "1", "2", "3", "4", "5" ] }, max_batch=2)
    responses = []
    for r in islice(m.sample("prompt", batch=3), 6):
        responses.append(r)
    assert responses == [ "0", "1", "2", "3", "4", "5" ]
    assert m.num_queries == 3

def test_batched_cached():
    m = MockBufferedModel({ "prompt": [ "0", "1", "2", "3", "4" ] }, max_batch=2)
    r = Repeatable(m)
    for s in islice(r.sample("prompt"), 2):
        pass
    responses = []
    start = m.num_queries
    for s in islice(r.sample("prompt", batch=2), 4):
        responses.append(s)
    assert responses == ["0", "1", "2", "3"]
    assert m.num_queries - start == 1
