"""
Example 06 — MLflow Model Registry loader

Demonstrates the full round-trip:
  1. Train a LogisticRegression classifier on iris data
  2. Log it to a local MLflow experiment
  3. Load it back via from_mlflow (no manual .load_model())
  4. Invoke as a standard ModelTool — OpenAI schema, direct invoke, confidence routing

Run:
    pip install predikit[mlflow]
    python examples/06_mlflow_loader.py
"""
import json
import tempfile

import mlflow
import mlflow.sklearn
import numpy as np
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from predikit import LowConfidenceError
from predikit.loaders import from_mlflow

# ---------------------------------------------------------------------------
# Step 1: Train and log to a local MLflow experiment
# ---------------------------------------------------------------------------
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=200).fit(X, y)

# Use a temp directory so this example doesn't litter the repo.
tmp_dir = tempfile.mkdtemp(prefix="predikit_mlflow_")
mlflow.set_tracking_uri(f"file://{tmp_dir}")
mlflow.set_experiment("predikit-demo")

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(clf, artifact_path="iris-classifier")
    run_id = run.info.run_id

model_uri = f"runs:/{run_id}/iris-classifier"
print(f"Logged model → {model_uri}\n")


# ---------------------------------------------------------------------------
# Step 2: Define the input schema
# ---------------------------------------------------------------------------
class IrisInput(BaseModel):
    sepal_length: float = Field(description="Sepal length in cm")
    sepal_width:  float = Field(description="Sepal width in cm")
    petal_length: float = Field(description="Petal length in cm")
    petal_width:  float = Field(description="Petal width in cm")


# ---------------------------------------------------------------------------
# Step 3: Load via from_mlflow — one call, no boilerplate
# ---------------------------------------------------------------------------
tool = from_mlflow(
    model_uri=model_uri,
    name="iris_classifier",
    description="Classify an iris flower: 0=setosa, 1=versicolor, 2=virginica.",
    input_schema=IrisInput,
    output_name="species",
    output_description="Predicted species index (0, 1, or 2)",
)

# ---------------------------------------------------------------------------
# Step 4: Use it exactly like any other ModelTool
# ---------------------------------------------------------------------------
sample = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}

print("=== Direct invoke ===")
print(tool.invoke(sample))
# → {"species": 0}

print("\n=== OpenAI function schema ===")
print(json.dumps(tool.to_openai(), indent=2))

# ---------------------------------------------------------------------------
# Step 5: Confidence routing — works unchanged because _PyfuncShim exposes
#         predict_proba() and classes_ from the underlying sklearn model.
# ---------------------------------------------------------------------------
confident_tool = from_mlflow(
    model_uri=model_uri,
    name="iris_confident",
    description="Iris classifier with confidence routing.",
    input_schema=IrisInput,
    output_name="species",
    output_description="Predicted species index",
    confidence_threshold=0.999,  # artificially high so warn path always fires
    on_low_confidence="warn",
)

print("\n=== Confidence routing (warn mode) ===")
result = confident_tool.invoke(sample)
if result.get("_low_confidence"):
    print(f"Low confidence ({result['_confidence']:.3f}) — agent should seek clarification")
else:
    print(result)
