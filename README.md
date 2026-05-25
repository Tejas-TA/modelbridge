# predikit
[![PyPI version](https://img.shields.io/pypi/v/predikit.svg)](https://pypi.org/project/predikit/)


Turn any trained scikit-learn or XGBoost model into an LLM-callable tool — auto-generated JSON schemas, typed I/O, zero boilerplate.

```python
tool = ModelTool(model=clf, name="classify_iris", ...)
tool.to_openai()              # OpenAI function schema, ready to pass to the API
tool.invoke({"sqft": 2200})   # → {"price_usd": 370730}
```

## Install

```bash
pip install predikit

# With XGBoost support
pip install predikit[xgboost]

# With LangChain support
pip install predikit[langchain]
```

## 30-second example

```python
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from predikit import ModelTool

# Train
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=200).fit(X, y)

# Define what the LLM will pass in
class IrisInput(BaseModel):
    sepal_length: float = Field(description="Sepal length in cm")
    sepal_width:  float = Field(description="Sepal width in cm")
    petal_length: float = Field(description="Petal length in cm")
    petal_width:  float = Field(description="Petal width in cm")

# Wrap the model
tool = ModelTool(
    model=clf,
    name="classify_iris",
    description="Classify an iris flower: 0=setosa, 1=versicolor, 2=virginica.",
    input_schema=IrisInput,
    output_name="species",
    output_description="Predicted species index",
)

# Get an OpenAI-ready schema
import json
print(json.dumps(tool.to_openai(), indent=2))

# Call it directly
tool.invoke({
    "sepal_length": 5.1, "sepal_width": 3.5,
    "petal_length": 1.4, "petal_width": 0.2,
})
# → {"species": 0}
```

## Core API

### `ModelTool`

```python
ModelTool(
    model,               # fitted sklearn-compatible estimator
    name: str,           # tool name the LLM sees
    description: str,    # tool description the LLM sees
    input_schema,        # Pydantic BaseModel describing inputs
    output_name: str,    # key for the prediction in the returned dict
    output_description: str,
)
```

| Method | Returns | What it does |
|--------|---------|--------------|
| `.invoke(input_dict)` | `dict` | Validates → predicts → returns `{output_name: value}` |
| `.to_openai()` | `dict` | OpenAI function-calling schema |
| `.to_langchain()` | `StructuredTool` | LangChain tool |
| `.to_callable()` | `Callable` | Plain Python function |

### `ToolRegistry`

Group multiple tools for bulk export:

```python
registry = ToolRegistry([price_tool, risk_tool])
registry.to_openai()     # → list[dict], pass directly to OpenAI
registry.to_langchain()  # → list[StructuredTool]
registry.get("name")     # → ModelTool
```

## Field naming rule

**Your Pydantic schema field names must exactly match the column names the model was trained on.**

predikit maps inputs to features by name, not position. If you trained on a DataFrame with columns `["sqft", "bedrooms"]`, your schema fields must be `sqft` and `bedrooms` — not `sq_ft`, not `Sqft`.

```python
# ✓ Columns match: sqft, bedrooms, bathrooms
class GoodInput(BaseModel):
    sqft:      float
    bedrooms:  float
    bathrooms: float

# ✗ Name mismatch — raises ValueError at runtime
class BadInput(BaseModel):
    square_footage: float  # model expects "sqft"
    beds:           float  # model expects "bedrooms"
    baths:          float  # model expects "bathrooms"
```

When there's a mismatch, predikit tells you exactly which names are wrong:

```
ValueError: Input schema is missing model features: ['sqft', 'bedrooms'].
Schema has: ['square_footage', 'beds', 'bathrooms'], model expects: ['sqft', 'bedrooms', 'bathrooms']
```

> **Tip:** If you trained with a numpy array (no DataFrame), predikit has no feature names to check — it uses your schema's field definition order instead.

## Cookbook

### XGBoost regression

```python
from xgboost import XGBRegressor
from predikit import ModelTool

reg = XGBRegressor().fit(X_train, y_train)

class HouseInput(BaseModel):
    sqft:       float
    bedrooms:   float
    year_built: float

tool = ModelTool(
    model=reg,
    name="price_estimate",
    description="Predict home price in USD.",
    input_schema=HouseInput,
    output_name="price_usd",
    output_description="Predicted sale price in USD",
)
```

### Multiple tools in one registry

```python
registry = ToolRegistry([price_tool, risk_tool, demand_tool])

# OpenAI
response = client.chat.completions.create(
    model="gpt-4o",
    tools=registry.to_openai(),
    ...
)

# LangChain
agent = initialize_agent(tools=registry.to_langchain(), ...)
```

### Bool inputs from an LLM

LLMs sometimes return `"yes"`, `"true"`, or `"1"` for boolean fields. predikit coerces these automatically before Pydantic validation:

```python
class Input(BaseModel):
    has_pool: bool

tool.invoke({"has_pool": "yes"})   # → coerced to True
tool.invoke({"has_pool": "false"}) # → coerced to False
tool.invoke({"has_pool": "maybe"}) # → raises ValueError with clear message
```

Supported strings: `true/false`, `yes/no`, `1/0`, `on/off`.

### Orlando real estate demo

See [`examples/03_orlando_real_estate.py`](examples/03_orlando_real_estate.py) for a full end-to-end walkthrough: synthetic dataset → XGBoost training → `ModelTool` → registry → OpenAI schema → prediction.

## Roadmap

Intentionally out of scope for v0.1 — planned for later releases:

- Confidence-aware routing & fallback
- Multi-model synthesis (agent calls several, reconciles results)
- MLflow / Snowflake Model Registry integration
- HuggingFace / PyTorch / TensorFlow support
- Async invocation

## License

MIT © Tejas Tumakuru Ashok
