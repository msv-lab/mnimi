# LLM Cache

An LLM cache stores responses from an LLM to avoid recomputation, making runs **reproducible**, **faster**, and **cheaper**, while enabling **deterministic debugging**.

⚠️ Caching changes the semantics of LLM calls. If not handled carefully, this can introduce subtle bugs.

## Caching Semantics

Without caching:
```python
r1 = model.sample(prompt)  # fresh sample
r2 = model.sample(prompt)  # fresh sample
```
Each call is independent.

With caching, two main modes are possible:

### Request-Level Caching

Request-level caching ensures for the same same prompt and model parameters you will always get the same cached result. Although it provides efficient optimizations and makes debugging easier, it introduces difficulties in implementing some common agentic patterns such as retries.

Example:
```python
model = AI302("gpt-4o", 1.0)
model = RequestLevelCache(model, "~/.llm_cache")

next(model.sample(prompt))  # "92"
next(model.sample(prompt))  # "92" (same result)

for s in islice(model.sample(prompt), 2):
    print(s) # "92" "747" (independent within sequence)
```

### Process-Level Independence

Process-level caching ensures per-call independence within a given process, thus mimicing live LLM API behavior (each `.sample` call is fresh).

Example:
```python
model = AI302("gpt-4o", 1.0)
model = RequestLevelCache(model, "~/.llm_cache")
model = IndependentSampling(model)  # ensure independence of samples within current process

next(model.sample(prompt))  # "92"
next(model.sample(prompt))  # "749" (different result)
```

## Hybrid Mode

The recommended way to use this cache is as follows:

1. Globally: identical requests share the same cached sequence.
2. Locally (when needed): iterating over `.sample(prompt)` yields fresh entries within that sequence.

Example with retries:

```python
model = AI302("gpt-4o", 1.0)
model = RequestLevelCache(model, "~/.llm_cache")

n1 = next(model.sample(prompt))  # "92"
n2 = next(model.sample(prompt))  # "92"

model = IndependentSampling(model)

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

To share only the subset of results for your current run:
```python
model = RequestLevelCache(model, "/path/to/original_cache/")
model = RequestLevelCache(model, "/path/to/sliced_cache")
run_experiment(model)
```
After execution, `sliced_cache` contains exactly what’s needed for reproducibility.


## Batch Sampling

Providers differ in max batch size. The cache API decouples provider limits from your experiment setup:
```python
model = CloseAI("gpt-4o", 1.0, max_batch=10)
model = RequestLevelCache(model, "~/.llm_cache")

for r in islice(model.sample(prompt, batch=20), 40):
    process(r)
```
This example queries the provider four times (splitting into batches automatically).

## Supported Providers

This code supports the following providers:

- 302.ai
- CloseAI
- FireworksAI

Since different providers have different names for the same model, the model's cache identifier can be changed by setting `alias`:

```
m = AI302("deepseek-v3-huoshan", 0.5, alias="deepseek-v3")

m = CloseAI("gpt-4o", 1.0)

m = FireworksAI("accounts/fireworks/models/llama-v3p1-8b-instruct", 1.0, alias="llama-3-8b")
```
