from __future__ import annotations
import asyncio
from collections import Counter
from typing import Any, Callable

from .tool import ModelTool
from .exporters.openai import to_openai_schema
from .exporters.langchain import to_langchain_tool

_VALID_STRATEGIES = {"collect", "mean", "weighted_mean", "vote", "weighted_vote"}


class ModelEnsemble:
    """Calls multiple ModelTools and reconciles their outputs into one dict."""

    def __init__(
        self,
        tools: list[ModelTool],
        name: str,
        description: str,
        strategy: str = "collect",
        weights: list[float] | None = None,
    ) -> None:
        if not tools:
            raise ValueError("ModelEnsemble requires at least one tool.")
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Use one of: {sorted(_VALID_STRATEGIES)}."
            )
        if weights is not None and len(weights) != len(tools):
            raise ValueError(
                f"weights length ({len(weights)}) must match number of tools ({len(tools)})."
            )
        self.tools = tools
        self.name = name
        self.description = description
        self.strategy = strategy
        self.weights = weights
        self.input_schema = tools[0].input_schema
        self.output_description = tools[0].output_description

    @property
    def output_name(self) -> str:
        return self.tools[0].output_name

    def invoke(self, input_dict: dict) -> dict:
        """Call all tools and reconcile outputs using the configured strategy."""
        return self._reconcile([t.invoke(input_dict) for t in self.tools])

    async def ainvoke(self, input_dict: dict) -> dict:
        """Call all tools concurrently and reconcile outputs using the configured strategy."""
        results = await asyncio.gather(*[t.ainvoke(input_dict) for t in self.tools])
        return self._reconcile(list(results))

    def _reconcile(self, results: list[dict]) -> dict:
        if self.strategy == "collect":
            merged: dict = {}
            for r in results:
                merged.update(r)
            return merged

        output_name = self.tools[0].output_name
        values = [r[output_name] for r in results]
        weights = self.weights if self.weights is not None else [1.0] * len(self.tools)

        if self.strategy == "mean":
            # Equal-weight average regardless of self.weights; use weighted_mean for non-uniform weights.
            numeric = [float(v) for v in values]
            return {output_name: sum(numeric) / len(numeric)}

        if self.strategy == "weighted_mean":
            numeric = [float(v) for v in values]
            total = sum(weights)
            return {output_name: sum(w * v for w, v in zip(weights, numeric)) / total}

        if self.strategy == "vote":
            return {output_name: Counter(values).most_common(1)[0][0]}

        if self.strategy == "weighted_vote":
            tally: dict[Any, float] = {}
            for w, v in zip(weights, values):
                tally[v] = tally.get(v, 0.0) + w
            return {output_name: max(tally, key=tally.__getitem__)}

        # unreachable — __init__ validates strategy — but satisfies type checkers
        raise ValueError(f"Unknown strategy '{self.strategy}'.")

    def to_openai(self) -> dict:
        return to_openai_schema(self)

    def to_langchain(self):
        return to_langchain_tool(self)

    def to_callable(self) -> Callable[..., dict]:
        def _fn(**kwargs) -> dict:
            return self.invoke(kwargs)
        _fn.__name__ = self.name
        _fn.__doc__ = self.description
        return _fn
