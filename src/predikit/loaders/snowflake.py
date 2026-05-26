from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from ..tool import ModelTool

try:
    from snowflake.ml.registry import Registry
except ImportError:
    Registry = None  # type: ignore[assignment,misc]


class _SnowflakeShim:
    """Wraps a Snowflake Model Registry model to look like an sklearn estimator."""

    def __init__(self, sf_model, output_method: str = "predict", classes=None) -> None:
        self._model = sf_model
        self._method = output_method
        if classes is not None:
            self.classes_ = classes

    def predict(self, X) -> np.ndarray:
        result = getattr(self._model, self._method)(X)
        if hasattr(result, "to_numpy"):
            return result.to_numpy().flatten()
        return np.asarray(result).flatten()


def from_snowflake(
    session,
    model_name: str,
    model_version: str,
    name: str,
    description: str,
    input_schema: type[BaseModel],
    output_name: str,
    output_description: str,
    output_method: str = "predict",
    classes=None,
    **model_tool_kwargs,
) -> ModelTool:
    """Load a registered Snowflake model and return it as a ModelTool.

    Args:
        session: An active ``snowflake.snowpark.Session``.
        model_name: Name of the model in the Snowflake Model Registry.
        model_version: Version string, e.g. ``"V3"``.
        name: Tool name the LLM sees.
        description: Tool description the LLM sees.
        input_schema: Pydantic BaseModel describing the model's inputs.
        output_name: Key for the prediction in the returned dict.
        output_description: Human-readable description of the output.
        output_method: Method to call on the registry model object (default ``"predict"``).
        classes: Optional list of class labels; enables confidence routing for classifiers.
        **model_tool_kwargs: Forwarded to ModelTool (e.g. ``confidence_threshold``).

    Returns:
        A fully configured :class:`~predikit.ModelTool`.
    """
    if Registry is None:
        raise ImportError(
            "snowflake-ml-python is required for from_snowflake(). "
            "Install it with: pip install predikit[snowflake]"
        )

    registry = Registry(session=session)
    sf_model = registry.get_model(model_name).version(model_version)
    shim = _SnowflakeShim(sf_model, output_method=output_method, classes=classes)
    return ModelTool(
        model=shim,
        name=name,
        description=description,
        input_schema=input_schema,
        output_name=output_name,
        output_description=output_description,
        **model_tool_kwargs,
    )
