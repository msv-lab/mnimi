from dataclasses import dataclass
from typing import Union, Any, Optional, List, Dict, Callable

from cached_llm import Model, Independent


class LLMOutputError(Exception):
    pass


@dataclass
class Match:
    start: int
    end: int
    value: Any


@dataclass(frozen=True)
class Seq:
    parts: List["OutputSpec"]

    def _match(self, text: str, pos: int) -> Match:
        cur = pos
        values: List[Any] = []
        spans: List[tuple[int, int]] = []

        for i, part in enumerate(self.parts):
            m = part._match(text, cur)
            values.append(m.value)
            spans.append((m.start, m.end))
            cur = m.end

        start = spans[0][0] if spans else pos
        end = spans[-1][1] if spans else pos
        return Match(start, end, values)
    

@dataclass(frozen=True)
class Tag:
    """HTML-style tag"""
    name: str

    def _match(self, text: str, pos: int) -> Match:
        open_tok = f"<{self.name}>"
        close_tok = f"</{self.name}>"

        start = text.find(open_tok, pos)
        if start == -1:
            raise LLMOutputError(f"Expected opening tag {open_tok} at {pos}")
        else:
            content_start = start + len(open_tok)
            close_pos = text.find(close_tok, content_start)
            if close_pos == -1:
                raise LLMOutputError(f"Expected closing tag {close_tok} at {content_start}")
            content = text[content_start:close_pos]
            end = close_pos + len(close_tok)
            return Match(start, end, content)


@dataclass(frozen=True)
class Code:
    """Markdown code block"""

    def _match(self, text: str, pos: int) -> Match:
        fence = "```"
        start = text.find(fence, pos)
        if start == -1:
            raise LLMOutputError(f"Expected markdown code fence ``` at {pos}")

        after_open = start + len(fence)
        nl = text.find("\n", after_open)
        if nl == -1:
            raise LLMOutputError(f"Unterminated code block (no newline after opening fence) at {after_open}")
        code_start = nl + 1

        end_fence = text.find("\n```", code_start)
        if end_fence == -1:
            # allow fence at very end without preceding newline:
            end_fence = text.find("```", code_start)
            if end_fence == -1:
                raise LLMOutputError(f"Unterminated code block (missing closing fence ```) at {code_start}")
            code_end = end_fence
            end = end_fence + len(fence)
        else:
            code_end = end_fence
            end = end_fence + 1 + len(fence)

        header = text[after_open:nl]  # language/info string (possibly empty)
        code = text[code_start:code_end]
        return Match(start, end, code)


def _match_rep(inner: "OutputSpec", text: str, pos: int, min_count: int) -> Match:
    """Note: specs like Seq([Star(Tag("a")), Code(), Tag("a")]) are not supported because of greedy parsing
    """
    cur = pos
    matches: List[Any] = []
    start0: Optional[int] = None
    last_end = pos

    while True:
        try:
            m = inner._match(text, cur)
        except LLMOutputError:
            break

        if m.end <= cur:
            break

        if start0 is None:
            start0 = m.start
        matches.append(m.value)
        last_end = m.end
        cur = m.end

    if len(matches) < min_count:
        raise LLMOutputError(f"Expected at least {min_count} occurrence(s), got {len(matches)} at {pos}")

    if start0 is None:
        start0 = pos
    return Match(start0, last_end, matches)
    

@dataclass(frozen=True)
class Star:
    """Kleene star"""
    inner: "OutputSpec"

    def _match(self, text: str, pos: int) -> Match:
        _match_rep(self, text, pos, min_count=0)


@dataclass(frozen=True)
class Plus:
    """Non-empty list"""
    inner: "OutputSpec"

    def _match(self, text: str, pos: int) -> Match:
        _match_rep(self, text, pos, min_count=1)
    

OutputSpec = Union[Seq, Tag, Code, Star, Plus]


def parse(spec: OutputSpec, text: str) -> Any:
    return spec._match(text, 0).value


def query_retry(model: Model, prompt: str, spec: OutputSpec, retries: int = 1,
                validator: Optional[Callable[[Any], bool]] = None) -> Any:
    model = Independent(model)

    for attempt in range(1, retries + 1):
        raw: Optional[str] = None
        try:
            raw = next(model.sample(prompt))
            parsed = parse(spec, raw)

            if validator is not None and not validator(parsed):
                raise ValueError("validator returned False")

            return parsed
        except Exception as e:
            if attempt == retries:
                msg = f"query_retry failed after {retries} attempts; last error: {type(e).__name__}: {e}"
                err = LLMOutputError(msg)
                raise err from e
