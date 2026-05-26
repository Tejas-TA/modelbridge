"""
Example 07 — Snowflake Model Registry loader

Demonstrates loading a registered Snowflake model as a ModelTool.
This example uses a mock session so it runs without real Snowflake credentials.
In production, replace mock_session with your live snowflake.snowpark.Session.

Run:
    pip install predikit
    python examples/07_snowflake_loader.py

For a real Snowflake connection:
    pip install predikit[snowflake]
    # Then replace mock_session with:
    #   from snowflake.snowpark import Session
    #   session = Session.builder.configs({...}).create()
"""
import json
from unittest.mock import MagicMock

import numpy as np
from pydantic import BaseModel, Field

from predikit.loaders.snowflake import _SnowflakeShim, from_snowflake

# ---------------------------------------------------------------------------
# Mock a Snowflake session and model (stand-in for real Snowpark objects)
# ---------------------------------------------------------------------------
mock_session = MagicMock()

# Simulate the model's predict output: probability of churn for each member.
mock_sf_model = MagicMock()
mock_sf_model.predict.return_value = np.array([0.73])


def _patch_registry(mock_sf_model):
    """Context manager that replaces Registry with a stub returning mock_sf_model."""
    from unittest.mock import patch
    registry_mock = MagicMock()
    registry_mock.get_model.return_value.version.return_value = mock_sf_model
    return patch("predikit.loaders.snowflake.Registry", return_value=registry_mock)


# ---------------------------------------------------------------------------
# Define the input schema matching the Snowflake model's feature columns
# ---------------------------------------------------------------------------
class MemberInput(BaseModel):
    tenure_months:  float = Field(description="Months as a Vacation Ownership member")
    trips_last_year: float = Field(description="Number of trips taken in the past 12 months")
    avg_spend:      float = Field(description="Average spend per trip in USD")


# ---------------------------------------------------------------------------
# Load via from_snowflake
# ---------------------------------------------------------------------------
with _patch_registry(mock_sf_model):
    tool = from_snowflake(
        session=mock_session,
        model_name="VACATION_CHURN",
        model_version="V3",
        name="churn_risk",
        description="Predict the probability that a Vacation Ownership member will churn.",
        input_schema=MemberInput,
        output_name="churn_probability",
        output_description="Churn probability in [0, 1]",
    )

# ---------------------------------------------------------------------------
# Invoke — works exactly like any ModelTool
# ---------------------------------------------------------------------------
print("=== Churn prediction ===")
result = tool.invoke({"tenure_months": 24.0, "trips_last_year": 2.0, "avg_spend": 500.0})
print(result)
# → {"churn_probability": 0.73}

print("\n=== OpenAI function schema ===")
print(json.dumps(tool.to_openai(), indent=2))

# ---------------------------------------------------------------------------
# predict_proba via output_method — pass "predict_proba" to surface class
# probabilities as the output, or use a custom scoring method.
# ---------------------------------------------------------------------------
mock_proba_model = MagicMock()
mock_proba_model.predict_proba.return_value = np.array([0.27])

with _patch_registry(mock_proba_model):
    proba_tool = from_snowflake(
        session=mock_session,
        model_name="VACATION_CHURN",
        model_version="V3",
        name="churn_proba",
        description="Churn probability from predict_proba.",
        input_schema=MemberInput,
        output_name="churn_probability",
        output_description="Probability of churn",
        output_method="predict_proba",
    )

print("\n=== Using output_method='predict_proba' ===")
print(proba_tool.invoke({"tenure_months": 6.0, "trips_last_year": 0.0, "avg_spend": 0.0}))
