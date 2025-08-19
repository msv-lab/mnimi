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
        self.num_sample_calls = 0
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
        self.num_sample_calls += 1
        i = MockModel._MockIterator(self, prompt)
        i.set_batch_size(batch)
        return i


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
    
