import asyncio
import numpy as np
import pytest
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from predikit import ModelTool


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@pytest.fixture
def iris_tool():
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(max_iter=200).fit(X, y)
    return ModelTool(
        model=clf,
        name="iris_classifier",
        description="Classify iris species from measurements",
        input_schema=IrisInput,
        output_name="species",
        output_description="Predicted iris species (0=setosa, 1=versicolor, 2=virginica)",
    )


def test_invoke_returns_dict(iris_tool):
    result = iris_tool.invoke(
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    )
    assert "species" in result
    assert result["species"] in [0, 1, 2]


def test_invoke_string_inputs_coerced(iris_tool):
    result = iris_tool.invoke(
        {"sepal_length": "5.1", "sepal_width": "3.5", "petal_length": "1.4", "petal_width": "0.2"}
    )
    assert result["species"] in [0, 1, 2]


def test_invoke_invalid_input_raises(iris_tool):
    with pytest.raises(ValueError):
        iris_tool.invoke({"sepal_length": "not_a_number"})


def test_to_callable(iris_tool):
    fn = iris_tool.to_callable()
    assert callable(fn)
    result = fn(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)
    assert "species" in result


def test_output_is_native_python_type(iris_tool):
    result = iris_tool.invoke(
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    )
    assert isinstance(result["species"], int)


def test_invoke_bool_string_coerced_through_invoke():
    class FlagInput(BaseModel):
        value: float
        active: bool

    X = np.array([[1.0, 0], [2.0, 1], [3.0, 0], [4.0, 1]])
    y = np.array([0, 1, 0, 1])
    clf = LogisticRegression().fit(X, y)

    tool = ModelTool(
        model=clf,
        name="flag_test",
        description="test bool coercion",
        input_schema=FlagInput,
        output_name="result",
        output_description="predicted class",
    )
    assert tool.invoke({"value": 3.0, "active": "yes"})["result"] in [0, 1]
    assert tool.invoke({"value": 1.0, "active": "false"})["result"] in [0, 1]
    assert tool.invoke({"value": 3.0, "active": "on"})["result"] in [0, 1]


SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


def test_ainvoke_returns_same_as_invoke(iris_tool):
    result = asyncio.run(iris_tool.ainvoke(SAMPLE))
    assert result == iris_tool.invoke(SAMPLE)


def test_ainvoke_invalid_input_raises(iris_tool):
    with pytest.raises(ValueError):
        asyncio.run(iris_tool.ainvoke({"sepal_length": "not_a_number"}))
