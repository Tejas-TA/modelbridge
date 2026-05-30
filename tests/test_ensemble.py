import asyncio
import pytest
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression

from predikit import ModelEnsemble, ModelTool, ToolRegistry

X, y = load_iris(return_X_y=True)
SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@pytest.fixture
def clf_tools():
    return [
        ModelTool(
            model=LogisticRegression(max_iter=200, C=c).fit(X, y),
            name=f"clf_C{c}", description="Classify iris",
            input_schema=IrisInput, output_name="species", output_description="species",
        )
        for c in [0.5, 1.0]
    ]


@pytest.fixture
def reg_tools():
    y_f = y.astype(float)
    return [
        ModelTool(
            model=LinearRegression().fit(X, y_f),
            name=f"reg_{i}", description="Regress iris",
            input_schema=IrisInput, output_name="value", output_description="value",
        )
        for i in range(2)
    ]


def test_collect_merges_distinct_output_names():
    clf = LogisticRegression(max_iter=200).fit(X, y)
    reg = LinearRegression().fit(X, y.astype(float))
    tool_a = ModelTool(model=clf, name="a", description="", input_schema=IrisInput, output_name="species", output_description="")
    tool_b = ModelTool(model=reg, name="b", description="", input_schema=IrisInput, output_name="score", output_description="")
    result = ModelEnsemble(tools=[tool_a, tool_b], name="e", description="", strategy="collect").invoke(SAMPLE)
    assert "species" in result
    assert "score" in result


def test_mean_averages_numeric_output(reg_tools):
    ensemble = ModelEnsemble(tools=reg_tools, name="mean_e", description="", strategy="mean")
    result = ensemble.invoke(SAMPLE)
    assert "value" in result
    # Two identical LinearRegression models → mean == either model's output
    single = reg_tools[0].invoke(SAMPLE)
    assert abs(result["value"] - single["value"]) < 1e-6


def test_vote_returns_majority(clf_tools):
    result = ModelEnsemble(tools=clf_tools, name="vote_e", description="", strategy="vote").invoke(SAMPLE)
    assert result["species"] in [0, 1, 2]


def test_invalid_strategy_raises(clf_tools):
    with pytest.raises(ValueError, match="Unknown strategy"):
        ModelEnsemble(tools=clf_tools, name="e", description="", strategy="bad").invoke(SAMPLE)


def test_empty_tools_raises():
    with pytest.raises(ValueError, match="at least one tool"):
        ModelEnsemble(tools=[], name="e", description="")


def test_to_openai_schema(clf_tools):
    ensemble = ModelEnsemble(tools=clf_tools, name="my_ensemble", description="test", strategy="vote")
    schema = ensemble.to_openai()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "my_ensemble"
    assert "parameters" in schema["function"]


def test_to_callable(clf_tools):
    fn = ModelEnsemble(tools=clf_tools, name="e", description="", strategy="vote").to_callable()
    assert callable(fn)
    result = fn(**SAMPLE)
    assert "species" in result


def test_ainvoke_returns_same_as_invoke_collect():
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(max_iter=200).fit(X, y)
    reg = LinearRegression().fit(X, y.astype(float))
    tool_a = ModelTool(model=clf, name="a", description="", input_schema=IrisInput, output_name="species", output_description="")
    tool_b = ModelTool(model=reg, name="b", description="", input_schema=IrisInput, output_name="score", output_description="")
    ens = ModelEnsemble(tools=[tool_a, tool_b], name="e", description="", strategy="collect")
    assert asyncio.run(ens.ainvoke(SAMPLE)) == ens.invoke(SAMPLE)


def test_ainvoke_returns_same_as_invoke_vote(clf_tools):
    ens = ModelEnsemble(tools=clf_tools, name="e", description="", strategy="vote")
    assert asyncio.run(ens.ainvoke(SAMPLE)) == ens.invoke(SAMPLE)


def test_ainvoke_returns_same_as_invoke_mean(reg_tools):
    ens = ModelEnsemble(tools=reg_tools, name="e", description="", strategy="mean")
    assert asyncio.run(ens.ainvoke(SAMPLE)) == ens.invoke(SAMPLE)


def test_registry_includes_ensembles(clf_tools):
    ensemble = ModelEnsemble(tools=clf_tools, name="vote_ens", description="", strategy="vote")
    registry = ToolRegistry(tools=clf_tools, ensembles=[ensemble])
    schemas = registry.to_openai()
    assert len(schemas) == len(clf_tools) + 1
    names = [s["function"]["name"] for s in schemas]
    assert "vote_ens" in names
