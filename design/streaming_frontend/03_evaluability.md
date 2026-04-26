# #7-design — Making the System Evaluable

## What changed

Four gaps were patched so an eval harness can run classifiers in isolation, record
what they produced, and trace every output back to the exact prompt that produced it.

1. **Prompt versioning** — every prompt module now exports `VERSION = "v1"` and a
   central `get_prompt_version(key)` function returns it.
2. **Stable artifact IDs** — `ThreadSegment` got a `thread_id` UUID (it was the only
   artifact without one).
3. **PromptKey integrity** — a silent enum alias bug was fixed. `PROFILE_SCANNER` was
   an invisible duplicate of `PROFILE_CLASSIFIER`.
4. **Dead import removed** — `from sympy.crypto.crypto import decipher_affine` in
   `session.py` was never used and was removed.

---

## Why this is the right call

Without these four properties, an eval harness is blind:

- **No versioning** → you can't tell which prompt produced a given output. If the model
  starts giving different results, you don't know if you changed the prompt or the model.
- **No stable IDs** → you can't join two pipeline runs to compare them. If run A
  produces a `ThreadSegment` and run B produces one too, which ones correspond?
- **Enum alias** → `list(PromptKey)` silently skipped `PROFILE_SCANNER`. Any loop over
  all keys (which an eval harness does) would silently skip the profile scanner.

These aren't theoretical — they are the exact reasons eval harnesses fail in practice.

---

## How it works

### Prompt versioning

Each prompt module exports a `VERSION` constant:

```python
# configure/prompts/intent_classifier.py
VERSION = "v1"
TEMPLATE = "..."
```

The prompt registry (`configure/prompts/__init__.py`) maps every `PromptKey` to its
module's VERSION:

```python
_VERSION_REGISTRY: dict[str, str] = {
    PromptKey.INTENT_CLASSIFIER.value: intent_classifier.VERSION,
    PromptKey.DECOMPOSER.value:        decomposer.VERSION,
    # ... all 11 keys
}

def get_prompt_version(key: str | PromptKey) -> str:
    lookup = key.value if isinstance(key, PromptKey) else key
    if lookup in _VERSION_REGISTRY:
        return _VERSION_REGISTRY[lookup]
    raise KeyError(f"Unknown prompt key {lookup!r}")
```

**Workflow**: when you make a meaningful change to a prompt, bump `VERSION = "v1"` to
`"v2"` in that module. The eval harness records `prompt_version` in every result record.
After the bump you can diff two run files and immediately see which outputs changed
alongside the version change.

### Stable artifact IDs

`ThreadSegment` was the only model without a stable UUID:

```python
# model/session.py
class ThreadSegment(BaseModel):
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # added
    thread_name: str
    exchange_ids: list[str]
    tags: list[Tag]
```

Now all three pipeline artifacts have consistent naming:
- `Exchange.exchange_id`
- `ThreadSegment.thread_id`
- `Fragment.fragment_id`

An eval harness can use these to join records: "show me all cases where this exchange
ended up in a thread with tag X in run A but tag Y in run B."

### The PromptKey alias bug

Python enums silently create **aliases** when two members share the same value:

```python
# BEFORE — broken
class PromptKey(Enum):
    PROFILE_CLASSIFIER = "profile_classifier"
    PROFILE_SCANNER    = "profile_classifier"   # same value → silent alias!
```

`list(PromptKey)` would return only `PROFILE_CLASSIFIER`. `PROFILE_SCANNER` existed
as `PromptKey.PROFILE_SCANNER` but was skipped in iteration. Any loop like
`for key in PromptKey: get_prompt_version(key)` would silently skip profile scanning.

The fix was to give `PROFILE_SCANNER` its own distinct value:

```python
# AFTER — correct
class PromptKey(Enum):
    PROFILE_SCANNER = "profile_scanner"   # own value, no longer an alias
```

And `PROFILE_CLASSIFIER` was removed entirely — it was the dead alias.

The `test_prompt_key_values_are_unique` test (`tests/test_evaluability.py`) permanently
guards against this class of bug:

```python
def test_prompt_key_values_are_unique():
    values = [key.value for key in PromptKey]
    assert len(values) == len(set(values))
```

---

## Before → After

### Before

```
configure/prompts/intent_classifier.py
    TEMPLATE = "..."            # no VERSION

model/session.py
    class PromptKey(Enum):
        PROFILE_CLASSIFIER = "profile_classifier"
        PROFILE_SCANNER    = "profile_classifier"  # silent alias

    class ThreadSegment(BaseModel):
        thread_name: str     # no stable ID
        exchange_ids: list[str]
        tags: list[Tag]
```

Evaluating the system meant: "run it, look at the output, compare by hand."

### After

```
configure/prompts/intent_classifier.py
    VERSION  = "v1"
    TEMPLATE = "..."

configure/prompts/__init__.py
    get_prompt_version(PromptKey.INTENT_CLASSIFIER)  # → "v1"

model/session.py
    class PromptKey(Enum):
        PROFILE_SCANNER = "profile_scanner"   # own value

    class ThreadSegment(BaseModel):
        thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        thread_name: str
        ...
```

Evaluating the system means: run the harness, save the JSONL, change a prompt, run
again, call `compare_runs(a, b)` and read the diff.

---

## Key things to remember

- `VERSION` is a plain string constant in each prompt module. Bump it when the prompt
  text changes in a way that would affect model behaviour. Don't bump for whitespace
  or comment changes.
- Python enum aliases are silent — the interpreter will not warn you. The uniqueness
  test is the only protection.
- Stable IDs matter for *joining* data across runs, not just within a single run. They
  let you say "this specific fragment in run A corresponds to this one in run B."
