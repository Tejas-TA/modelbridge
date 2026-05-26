from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from ..tool import ModelTool


class _PyfuncShim:
    """Wraps an mlflow.pyfunc model to look like an sklearn estimator."""

    def __init__(self, pyfunc_model, classes=None) -> None:
        self._model = pyfunc_model

        # Surface sklearn metadata (feature names, n_features, classes) so
        # ModelTool's introspect() and _to_array() work without any extra config.
        impl = getattr(pyfunc_model, "_model_impl", None)
        underlying = None
        if impl is not None:
            underlying = getattr(impl, "sklearn_model", None)

        # Auto-detect classes from the underlying model if not supplied.
        if classes is None and underlying is not None and hasattr(underlying, "classes_"):
            classes = list(underlying.classes_)

        if classes is not None:
            self.classes_ = classes

        if underlying is not None:
            if hasattr(underlying, "feature_names_in_"):
                self.feature_names_in_ = underlying.feature_names_in_
            if hasattr(underlying, "n_features_in_"):
                self.n_features_in_ = underlying.n_features_in_

    def predict(self, X) -> np.ndarray:
        result = self._model.predict(X)
        if hasattr(result, "to_numpy"):
            return result.to_numpy().flatten()
        return np.asarray(result).flatten()

    def predict_proba(self, X) -> np.ndarray:
        impl = getattr(self._model, "_model_impl", None)
        if impl is not None:
            underlying = getattr(impl, "sklearn_model", None)
            if underlying is not None and hasattr(underlying, "predict_proba"):
                return underlying.predict_proba(X)
        raise NotImplementedError(
            "predict_proba is not available for this MLflow pyfunc model. "
            "The underlying model may not expose it through the pyfunc interface."
        )


def from_mlflow(
    model_uri: str,
    name: str,
    description: str,
    input_schema: type[BaseModel],
    output_name: str,
    output_description: str,
    **model_tool_kwargs,
) -> ModelTool:
    """Load a registered MLflow model and return it as a ModelTool.

    Args:
        model_uri: MLflow model URI, e.g. ``"models:/my-model/Production"``.
        name: Tool name the LLM sees.
        description: Tool description the LLM sees.
        input_schema: Pydantic BaseModel describing the model's inputs.
        output_name: Key for the prediction in the returned dict.
        output_description: Human-readable description of the output.
        **model_tool_kwargs: Forwarded to ModelTool (e.g. ``confidence_threshold``).

    Returns:
        A fully configured :class:`~predikit.ModelTool`.
    """
    try:
        import mlflow.pyfunc
    except ImportError as exc:
        raise ImportError(
            "mlflow is required for from_mlflow(). Install it with: pip install predikit[mlflow]"
        ) from exc

    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    shim = _PyfuncShim(pyfunc_model)
    return ModelTool(
        model=shim,
        name=name,
        description=description,
        input_schema=input_schema,
        output_name=output_name,
        output_description=output_description,
        **model_tool_kwargs,
    )
