from structured_output import (
    Sequence, Tag, Code, Repeat,
    parse
)


def test_simple():
    spec = Sequence([
        Tag("analysis"),
        Tag("final"),
        Code(),
    ])

    out = """Some preface text

<analysis>reasoning...</analysis>

random chatter

<final>answer</final>

more stuff

```python
print("hi")
tail text
```
"""
    result = parse(spec, out)
    
    assert result[0] == "reasoning..."
    assert result[1] == "answer"
    assert result[2] == 'print("hi")\ntail text'
