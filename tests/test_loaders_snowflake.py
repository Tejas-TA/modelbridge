"""Unit tests for from_snowflake: fully mocked — no Snowflake connection needed."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import BaseModel, Field

from predikit.loaders.snowflake import _SnowflakeShim, from_snowflake


# ---------------------------------------------------------------------------
# Shared schema
# ---------------------------------------------------------------------------

class MemberInput(BaseModel):
    tenure_months: float = Field(description="Months as a member")
    trips_last_year: float = Field(description="Trips taken in past 12 months")
    avg_spend: float = Field(description="Average spend per trip in USD")


# ---------------------------------------------------------------------------
# _SnowflakeShim unit tests
# ---------------------------------------------------------------------------

class TestSnowflakeShim:
    def test_predict_calls_correct_method(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        shim = _SnowflakeShim(mock_model)
        result = shim.predict(np.array([[24.0, 2.0, 500.0]]))
        mock_model.predict.assert_called_once()
        assert result[0] == 1

    def test_custom_output_method(self):
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.27, 0.73]])
        shim = _SnowflakeShim(mock_model, output_method="predict_proba")
        result = shim.predict(np.array([[24.0, 2.0, 500.0]]))
        mock_model.predict_proba.assert_called_once()
        assert result[0] == pytest.approx(0.27, abs=1e-6)

    def test_classes_optional(self):
        shim_no_classes = _SnowflakeShim(MagicMock())
        assert not hasattr(shim_no_classes, "classes_")

        shim_with_classes = _SnowflakeShim(MagicMock(), classes=[0, 1])
        assert shim_with_classes.classes_ == [0, 1]

    def test_predict_flattens_dataframe_result(self):
        import pandas as pd
        mock_model = MagicMock()
        mock_model.predict.return_value = pd.DataFrame({"output": [1]})
        shim = _SnowflakeShim(mock_model)
        result = shim.predict(np.array([[1.0, 2.0, 3.0]]))
        assert result.ndim == 1
        assert result[0] == 1

    def test_predict_flattens_numpy_result(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.73]])
        shim = _SnowflakeShim(mock_model)
        result = shim.predict(np.array([[1.0]]))
        assert result.ndim == 1
        assert result[0] == pytest.approx(0.73)


# ---------------------------------------------------------------------------
# from_snowflake integration tests (mocked session)
# ---------------------------------------------------------------------------

def _make_mock_session_and_model(return_value):
    """Return (mock_session, mock_sf_model) wired up through a mock Registry."""
    mock_session = MagicMock()
    mock_sf_model = MagicMock()
    mock_sf_model.predict.return_value = return_value
    return mock_session, mock_sf_model


class TestFromSnowflake:
    def test_roundtrip_invoke(self):
        mock_session, mock_sf_model = _make_mock_session_and_model(np.array([1]))

        with patch("predikit.loaders.snowflake.Registry") as MockRegistry:
            MockRegistry.return_value.get_model.return_value.version.return_value = mock_sf_model

            tool = from_snowflake(
                session=mock_session,
                model_name="VACATION_CHURN",
                model_version="V3",
                name="churn_risk",
                description="Predict member churn.",
                input_schema=MemberInput,
                output_name="churn_class",
                output_description="Predicted churn class",
            )

        result = tool.invoke({"tenure_months": 24.0, "trips_last_year": 2.0, "avg_spend": 500.0})
        assert result == {"churn_class": 1}

    def test_registry_called_with_correct_args(self):
        mock_session, mock_sf_model = _make_mock_session_and_model(np.array([0]))

        with patch("predikit.loaders.snowflake.Registry") as MockRegistry:
            mock_registry = MockRegistry.return_value
            mock_registry.get_model.return_value.version.return_value = mock_sf_model

            from_snowflake(
                session=mock_session,
                model_name="VACATION_CHURN",
                model_version="V3",
                name="churn_risk",
                description="Predict churn.",
                input_schema=MemberInput,
                output_name="churn_class",
                output_description="Churn class",
            )

            MockRegistry.assert_called_once_with(session=mock_session)
            mock_registry.get_model.assert_called_once_with("VACATION_CHURN")
            mock_registry.get_model.return_value.version.assert_called_once_with("V3")

    def test_custom_output_method_forwarded(self):
        mock_session = MagicMock()
        mock_sf_model = MagicMock()
        mock_sf_model.score.return_value = np.array([0.87])

        with patch("predikit.loaders.snowflake.Registry") as MockRegistry:
            MockRegistry.return_value.get_model.return_value.version.return_value = mock_sf_model

            tool = from_snowflake(
                session=mock_session,
                model_name="SCORE_MODEL",
                model_version="V1",
                name="scorer",
                description="Score members.",
                input_schema=MemberInput,
                output_name="score",
                output_description="Member score",
                output_method="score",
            )

        result = tool.invoke({"tenure_months": 12.0, "trips_last_year": 5.0, "avg_spend": 300.0})
        assert result["score"] == pytest.approx(0.87, abs=1e-6)
        mock_sf_model.score.assert_called_once()

    def test_model_tool_kwargs_forwarded(self):
        mock_session, mock_sf_model = _make_mock_session_and_model(np.array([1]))

        with patch("predikit.loaders.snowflake.Registry") as MockRegistry:
            MockRegistry.return_value.get_model.return_value.version.return_value = mock_sf_model

            tool = from_snowflake(
                session=mock_session,
                model_name="CHURN",
                model_version="V1",
                name="churn",
                description="Churn.",
                input_schema=MemberInput,
                output_name="churn_class",
                output_description="Churn class",
                classes=[0, 1],
                confidence_threshold=0.85,
                on_low_confidence="warn",
            )

        assert tool.confidence_threshold == 0.85
        assert tool.on_low_confidence == "warn"

    def test_import_error_without_snowflake_ml(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "snowflake", None)
        monkeypatch.setitem(sys.modules, "snowflake.ml", None)
        monkeypatch.setitem(sys.modules, "snowflake.ml.registry", None)

        import importlib
        import predikit.loaders.snowflake as sf_mod
        importlib.reload(sf_mod)

        with pytest.raises(ImportError, match="snowflake-ml-python is required"):
            sf_mod.from_snowflake(
                session=MagicMock(),
                model_name="X",
                model_version="V1",
                name="x",
                description="x",
                input_schema=MemberInput,
                output_name="y",
                output_description="y",
            )
