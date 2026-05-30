from __future__ import annotations
import asyncio
import time
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel

from .introspect import introspect
from .coerce import coerce_inputs, coerce_value
from .exporters.openai import to_openai_schema
from .exporters.langchain import to_langchain_tool
from .exceptions import LowConfidenceError

_VALID_ON_LOW_CONFIDENCE = {"warn", "raise", "fallback"}
_SCALAR_TYPES = (bool, int, float, str)


class ModelTool:
    """Wraps a fitted sklearn-compatible model as an LLM-callable tool."""

    def __init__(
        self,
        model: Any,
        name: str,
        description: str,
        input_schema: type[BaseModel],
        output_name: str,
        output_description: str,
        confidence_threshold: float | None = None,
        on_low_confidence: str = "warn",
        fallback_tool: ModelTool | None = None,
        verbose: bool = False,
    ) -> None:
        if on_low_confidence not in _VALID_ON_LOW_CONFIDENCE:
            raise ValueError(f"on_low_confidence must be one of {_VALID_ON_LOW_CONFIDENCE}, got {on_low_confidence!r}")
        self.model = model
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.output_name = output_name
        self.output_description = output_description
        self._meta = introspect(model)
        self.confidence_threshold = confidence_threshold
        self.on_low_confidence = on_low_confidence
        self.fallback_tool = fallback_tool
        self.verbose = verbose

    def invoke(self, input_dict: dict) -> dict:
        """Validate inputs, run prediction, return {output_name: value}."""
        if self.verbose:
            print(f"[predikit] Invoking tool: {self.name}")
            print(f"[predikit] Input dict: {input_dict}")

        pre_coerced = self._pre_coerce(input_dict)
        try:
            validated = self.input_schema(**pre_coerced)
        except Exception as exc:
            raise ValueError(f"Input validation failed for '{self.name}': {exc}") from exc

        features = coerce_inputs(validated, self._meta)
        X = self._to_array(features)

        start_time = time.perf_counter()
        prediction = self.model.predict(X)[0]
        duration = (time.perf_counter() - start_time) * 1000

        if hasattr(prediction, "item"):
            prediction = prediction.item()

        if self.verbose:
            print(f"[predikit] Prediction: {prediction}")
            print(f"[predikit] Latency: {duration:.2f}ms")

        if (
            self.confidence_threshold is not None
            and self._meta["task"] == "classification"
            and hasattr(self.model, "predict_proba")
        ):
            proba = self.model.predict_proba(X)[0]
            confidence = float(max(proba))
            if confidence < self.confidence_threshold:
                return self._handle_low_confidence(input_dict, prediction, confidence)

        return {self.output_name: prediction}

    async def ainvoke(self, input_dict: dict) -> dict:
        """Async version of invoke(); runs the blocking call in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.invoke, input_dict)

    def to_openai(self) -> dict:
        """Return an OpenAI function-calling schema dict."""
        return to_openai_schema(self)

    def to_langchain(self):
        """Return a LangChain StructuredTool."""
        return to_langchain_tool(self)

    def to_callable(self) -> Callable[..., dict]:
        """Return a plain Python function that calls invoke()."""
        def _fn(**kwargs) -> dict:
            return self.invoke(kwargs)
        _fn.__name__ = self.name
        _fn.__doc__ = self.description
        return _fn

    def _handle_low_confidence(self, input_dict: dict, prediction: Any, confidence: float) -> dict:
        if self.on_low_confidence == "raise":
            raise LowConfidenceError(
                f"'{self.name}' confidence {confidence:.3f} is below threshold {self.confidence_threshold:.3f}"
            )
        if self.on_low_confidence == "fallback" and self.fallback_tool is not None:
            return self.fallback_tool.invoke(input_dict)
        return {self.output_name: prediction, "_confidence": confidence, "_low_confidence": True}

    def _pre_coerce(self, input_dict: dict) -> dict:
        # Runs coerce_value before Pydantic so LLM strings like "yes"/"no" work for bool fields.
        fields = self.input_schema.model_fields
        result = {}
        for k, v in input_dict.items():
            field = fields.get(k)
            annotation = getattr(field, "annotation", None) if field else None
            if annotation in _SCALAR_TYPES:
                try:
                    result[k] = coerce_value(v, annotation)
                except (ValueError, TypeError):
                    result[k] = v
            else:
                result[k] = v
        return result

    def _to_array(self, features: list) -> Any:
        feature_names = self._meta.get("feature_names")
        if feature_names:
            try:
                import pandas as pd
                return pd.DataFrame([dict(zip(feature_names, features))])
            except ImportError:
                pass
        return np.array([features])
