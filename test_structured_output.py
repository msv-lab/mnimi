from structured_output import (
    Seq, Tag, Code, Star, Plus,
    parse
)


def test_simple():
    spec = Seq([
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
