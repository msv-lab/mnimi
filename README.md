# LLM Cache

An LLM cache stores responses from an LLM to avoid recomputation, making runs reproducible, faster, and cheaper, while also enabling deterministic debugging.

This is a single-file, dependency-free implementation that provides a single API function `sample(prompt: str, batch: int = 1) -> Iterator[str]`. To use it, just copy the file `cached_llm.py` to your project.

## Independent, Repeatable and Persistent Sampling

Without caching, each call of `sample` returns an independent value:

```python
model = AI302("gpt-4o", 1.0)

prompt = "Choose a number between 1 and 1000."

n1 = next(model.sample(prompt))  # "92" 
n2 = next(model.sample(prompt))  # "747"
```

Using `Repeatable` (in-memory cache), you can change the semantics of `sample` so that for each call it returns exactly the same sequence:

```
model = Repeatable(model)

n3 = next(model.sample(prompt))  # "131" 
n4 = next(model.sample(prompt))  # "131" (same result)

for s in islice(model.sample(prompt), 2):
    print(s)  # "131" "561" (independent within sequence)
```

After that, the model can again be turned into independent sampling using `Independent`:

```
model = Independent(model)

n5 = next(model.sample(prompt))  # "131" 
n6 = next(model.sample(prompt))  # "561"
```

Note that it will still take values from the in-memory cache from the underying model.

`Persistent` is like `Repeatable`, but it persists across runs of your program:

Example:
```python
model = AI302("gpt-4o", 1.0)
model = Persistent(model, "~/.llm_cache")

prompt = "Choose a number between 1 and 1000."

n1 = next(model.sample(prompt))  # "92"
n2 = next(model.sample(prompt))  # "92" (same result across runs)
```

## Usage Pattern

The recommended way to use this cache is as follows:

1. Most of your code should not rely on whether your `sample` is independent or repeatable/persistent.
2. If your function relies on independence, it should start with `model = Independent(model)`.
3. If your function relies on repeatability, it should start with `model = Repeatable(model)`.

This example combines persistent caching with retries, which require independent sampling:

```python
model = AI302("gpt-4o", 1.0)
model = Persistent(model, "~/.llm_cache")

...

model = Independent(model)

for attempt in range(NUM_RETRIES):
	try:
	    x = f(model)
	    y = g(model, x)
	    z = t(model, y)
	    break
	except BadResponse:
	    if attempt == NUM_RETRIES - 1:
			raise BadResponse
		pass
```

## Cache Slicing

To extract only the subset of cache for your current run, use nested cache:

```python
model = Persistent(model, "/path/to/original_cache/")
model = Persistent(model, "/path/to/sliced_cache")
run_experiment(model)
```

After execution, `sliced_cache` contains exactly what’s needed for reproducibility.


## Batch Sampling

Providers differ in max batch size. This API decouples provider limits from your experiment setup:

```python
model = CloseAI("gpt-4o", 1.0, max_batch=10)
model = Persistent(model, "~/.llm_cache")

for r in islice(model.sample(prompt, batch=20), 40):
    process(r)
```

In this example, because the provider allows only 10 samples per request while the algorithm needs 40 in total (ideally in batches of 20), the code will automatically split the work into four requests of 10 samples each.


## Supported Providers

This code supports OpenAI-compatile HTTP API, with convenience wrappers for the following providers:

- 302.ai
- CloseAI
- FireworksAI
- XMCP

Since different providers have different names for the same model, the model's cache identifier can be changed by setting `alias`:

```
m = AI302("deepseek-v3-huoshan", 0.5, alias="deepseek-v3")

m = CloseAI("gpt-4o", 1.0)

m = FireworksAI("accounts/fireworks/models/llama-v3p1-8b-instruct", 1.0, alias="llama-3.1-8b")
```
