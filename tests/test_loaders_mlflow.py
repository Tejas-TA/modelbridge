"""Integration tests for from_mlflow: logs a real model, loads, invokes."""
import os
import tempfile

import numpy as np
import pytest
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

mlflow = pytest.importorskip("mlflow", reason="mlflow not installed")

from predikit.loaders import from_mlflow
from predikit.loaders.mlflow import _PyfuncShim


# ---------------------------------------------------------------------------
# Shared schemas
# ---------------------------------------------------------------------------

class IrisInput(BaseModel):
    sepal_length: float = Field(description="Sepal length in cm")
    sepal_width: float = Field(description="Sepal width in cm")
    petal_length: float = Field(description="Petal length in cm")
    petal_width: float = Field(description="Petal width in cm")


class RegressionInput(BaseModel):
    x0: float
    x1: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_sklearn_model(model, tmp_path):
    """Log a model to a local MLflow run and return its URI."""
    import mlflow.sklearn
    tracking_uri = tmp_path.as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id
    return f"runs:/{run_id}/model"


# ---------------------------------------------------------------------------
# _PyfuncShim unit tests
# ---------------------------------------------------------------------------

class TestPyfuncShim:
    def test_predict_returns_flat_array(self, tmp_path):
        X, y = load_iris(return_X_y=True)
        clf = LogisticRegression(max_iter=200).fit(X, y)
        uri = _log_sklearn_model(clf, tmp_path)

        import mlflow.pyfunc
        pyfunc_model = mlflow.pyfunc.load_model(uri)
        shim = _PyfuncShim(pyfunc_model)

        result = shim.predict(np.array([[5.1, 3.5, 1.4, 0.2]]))
        assert result.ndim == 1
        assert result[0] in clf.classes_

    def test_auto_detects_classes(self, tmp_path):
        X, y = load_iris(return_X_y=True)
        clf = LogisticRegression(max_iter=200).fit(X, y)
        uri = _log_sklearn_model(clf, tmp_path)

        import mlflow.pyfunc
        pyfunc_model = mlflow.pyfunc.load_model(uri)
        shim = _PyfuncShim(pyfunc_model)

        # Should have auto-detected classes_ from the underlying sklearn model.
        assert hasattr(shim, "classes_")
        assert list(shim.classes_) == [0, 1, 2]

    def test_predict_proba_delegates_to_underlying(self, tmp_path):
        X, y = load_iris(return_X_y=True)
        clf = LogisticRegression(max_iter=200).fit(X, y)
        uri = _log_sklearn_model(clf, tmp_path)

        import mlflow.pyfunc
        pyfunc_model = mlflow.pyfunc.load_model(uri)
        shim = _PyfuncShim(pyfunc_model)

        proba = shim.predict_proba(np.array([[5.1, 3.5, 1.4, 0.2]]))
        assert proba.shape == (1, 3)
        assert abs(proba[0].sum() - 1.0) < 1e-6

    def test_no_classes_for_regression(self, tmp_path):
        X = np.random.default_rng(0).random((50, 2))
        y = X[:, 0] * 2 + X[:, 1]
        reg = DecisionTreeRegressor().fit(X, y)
        uri = _log_sklearn_model(reg, tmp_path)

        import mlflow.pyfunc
        pyfunc_model = mlflow.pyfunc.load_model(uri)
        shim = _PyfuncShim(pyfunc_model)

        assert not hasattr(shim, "classes_")


# ---------------------------------------------------------------------------
# from_mlflow integration tests
# ---------------------------------------------------------------------------

class TestFromMlflow:
    def test_roundtrip_classifier(self, tmp_path):
        X, y = load_iris(return_X_y=True)
        clf = LogisticRegression(max_iter=200).fit(X, y)
        uri = _log_sklearn_model(clf, tmp_path)

        tool = from_mlflow(
            model_uri=uri,
            name="iris_classifier",
            description="Classify iris species.",
            input_schema=IrisInput,
            output_name="species",
            output_description="Predicted species index",
        )

        result = tool.invoke({
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        })
        assert "species" in result
        assert result["species"] in [0, 1, 2]

    def test_roundtrip_regressor(self, tmp_path):
        rng = np.random.default_rng(42)
        X = rng.random((100, 2))
        y = X[:, 0] * 3.0 + X[:, 1] * 1.5
        reg = DecisionTreeRegressor().fit(X, y)
        uri = _log_sklearn_model(reg, tmp_path)

        tool = from_mlflow(
            model_uri=uri,
            name="value_estimator",
            description="Estimate value.",
            input_schema=RegressionInput,
            output_name="value",
            output_description="Predicted value",
        )

        result = tool.invoke({"x0": 0.5, "x1": 0.5})
        assert "value" in result
        assert isinstance(result["value"], float)

    def test_model_tool_kwargs_forwarded(self, tmp_path):
        X, y = load_iris(return_X_y=True)
        clf = LogisticRegression(max_iter=200).fit(X, y)
        uri = _log_sklearn_model(clf, tmp_path)

        tool = from_mlflow(
            model_uri=uri,
            name="iris_confident",
            description="Iris with confidence routing.",
            input_schema=IrisInput,
            output_name="species",
            output_description="Predicted species index",
            confidence_threshold=0.999,
            on_low_confidence="warn",
        )

        assert tool.confidence_threshold == 0.999
        assert tool.on_low_confidence == "warn"

    def test_import_error_without_mlflow(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "mlflow", None)
        monkeypatch.setitem(sys.modules, "mlflow.pyfunc", None)

        # Force re-import so the ImportError path fires.
        import importlib
        import predikit.loaders.mlflow as loader_mod
        importlib.reload(loader_mod)

        with pytest.raises(ImportError, match="mlflow is required"):
            loader_mod.from_mlflow(
                model_uri="models:/fake/Production",
                name="x",
                description="x",
                input_schema=IrisInput,
                output_name="y",
                output_description="y",
            )
