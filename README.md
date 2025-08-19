# LLM Cache for Agentic Workflows

LLM cache stores responses from an LLM to avoid redundant queries, making runs reproducible, faster, and cheaper, while also enabling deterministic debugging.

This implementation:

- Is single-file & dependency-free; to use it, just copy `cached_llm.py` to your project.
- Provides a single API function `sample(prompt: str, batch: int = 1) -> Iterator[str]`.
- Supports agentic workflows like retries that conflict with naive caching.
- Supports batch sampling (getting multiple responses for a single HTTP request).
- Supports cache slicing (extracting relevant parts of cache).
- Supports multiple providers (302.ai, Fireworks, CloseAI, XMCP, etc).


## Independent, Repeatable and Persistent Sampling

Without caching, each call of `sample` returns an independent value:

```python
model = AI302("gpt-4o", 1.0)

prompt = "Choose a number between 1 and 1000."

n1 = next(model.sample(prompt))  # "92"
n2 = next(model.sample(prompt))  # "747"
```

Using `Repeatable` (in-memory cache), you can change the semantics of `sample` so that for each call it returns exactly the same sequence:

```python
model = Repeatable(model)

n3 = next(model.sample(prompt))  # "131" 
n4 = next(model.sample(prompt))  # "131" - same result

for s in islice(model.sample(prompt), 2):
    print(s)  # "131" "561" - independent within sequence
```

After that, the model can again be turned into independent sampling using `Independent`:

```python
model = Independent(model)

n5 = next(model.sample(prompt))  # "131" 
n6 = next(model.sample(prompt))  # "561"
```

Note that it will still take values from the underlying in-memory cache.

`Persistent` is like `Repeatable`, but it persists across runs of your program:

```python
model = AI302("gpt-4o", 1.0)
model = Persistent(model, "~/.llm_cache")

prompt = "Choose a number between 1 and 1000."

n1 = next(model.sample(prompt))  # "92"
n2 = next(model.sample(prompt))  # "92" - same result across runs
```

## Agentic Workflow Patterns

The recommended way to use this cache is as follows:

1. Use a repeatable/persistent model by default for maximum efficiency and determinism.
2. Most of your code should not rely on whether model is independent or repeatable.
3. If your logic relies on independence, start it with `model = Independent(model)`.
4. If your logic relies on repeatability, start it with `model = Repeatable(model)`.

This example combines persistent caching with retries, which require independent sampling:

```python
model = AI302("gpt-4o", 1.0)
model = Persistent(model, "~/.llm_cache")

ind_model = Independent(model)

for attempt in range(NUM_RETRIES):
	try:
        rep_model = Repeatable(ind_model)
	    x = step_1(rep_model)
	    y = step_2(rep_model, x)
	    z = step_3(rep_model, y)
	    break
	except Exception:
	    if attempt == NUM_RETRIES - 1:
			raise Exception("did not get good response")
		pass
```

Here, the calls of `sample` are independent across attempts, but repeat across the calls of `step_1`, `step_2` and `step_3` within each individual attempt.


## Cache Slicing

Cache slicing lets you extract a minimal subset of a larger cache for sharing just what's needed for replication. To achieve that, use nested caches:

```python
model = Persistent(model, "/path/to/original_cache/")
model = Persistent(model, "/path/to/sliced_cache")
run_experiment(model)
```

After execution, `sliced_cache` contains exactly what was used during this run.


## Batch Sampling

Providers differ in max batch size. This API decouples provider limits from your experiment setup:

```python
model = CloseAI("gpt-4o", 1.0, max_batch=10),
model = Persistent(model, "~/.llm_cache")

for r in islice(model.sample(prompt, batch=20), 40):
    process(r)
```

In this example, because the provider allows only 10 samples per request while the algorithm needs 40 in total (ideally in batches of 20), the code will automatically split the work into four requests of 10 samples each.


## Replication Mode

To ensure that your run relies only on cache (without quering the underlying model), use `Persistent(model, cache_dir, replication=True)`. In case of a cache miss, `sample` will raise `ReplicationCacheMiss`.


## Supported Providers

This code supports OpenAI-compatile HTTP API, with convenience wrappers for the following providers:

| Provider | API key environment variable |
| -------- | ------- |
| 302.ai | `AI302_API_KEY` |
| CloseAI | `CLOSEAI_API_KEY` |
| FireworksAI | `FIREWORKS_API_KEY` |
| XMCP | `XMCP_API_KEY` |
| OpenAI Compatible | set in code |

Since different providers have different names for the same model, the model's cache identifier can be changed by setting `alias`:

```python
models = [
    CloseAI("gpt-4o", 1.0),
    AI302("deepseek-v3-huoshan", 0.5, alias="deepseek-v3"),
    FireworksAI("accounts/fireworks/models/llama-v3p1-8b-instruct", 1.0, alias="llama-3.1-8b"),
    XMCP("ali/qwen2.5-7b-instruct", 1.0, alias="qwen2.5-7b")
]
```
