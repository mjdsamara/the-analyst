"""Tests for the Modeling Agent."""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Set environment variables before importing modules that use Settings
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-testing")
os.environ.setdefault("HUGGINGFACE_TOKEN", "test-token")

from src.agents.modeling import (
    DataSplit,
    FeatureImportance,
    ForecastResult,
    ModelDiagnostics,
    ModelingAgent,
    ModelingOutput,
    ModelMetrics,
    ModelResult,
    ModelType,
    PredictionResult,
    TaskType,
)


class TestModelingAgent:
    """Test suite for ModelingAgent."""

    @pytest.fixture
    def agent(self):
        """Create a modeling agent for testing."""
        with patch("src.agents.base.AsyncAnthropic"):
            return ModelingAgent()

    @pytest.fixture
    def regression_df(self):
        """Create a sample DataFrame for regression testing."""
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        # y = 2*x1 + 3*x2 + noise
        y = 2 * x1 + 3 * x2 + np.random.normal(0, 0.5, n)
        return pd.DataFrame({"feature1": x1, "feature2": x2, "target": y})

    @pytest.fixture
    def classification_df(self):
        """Create a sample DataFrame for classification testing."""
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        # Binary classification based on x1 + x2
        y = (x1 + x2 > 0).astype(int)
        return pd.DataFrame({"feature1": x1, "feature2": x2, "label": y})

    @pytest.fixture
    def time_series_df(self):
        """Create a sample DataFrame for time series testing."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        # Simple trend + noise
        values = np.arange(100) * 0.5 + np.random.normal(0, 2, 100) + 100
        return pd.DataFrame({"date": dates, "value": values})

    def test_agent_initialization(self, agent):
        """Test that agent initializes correctly."""
        assert agent.name == "modeling"
        assert agent.autonomy.value == "advisory"
        assert agent._random_state == 42

    @pytest.mark.asyncio
    async def test_execute_regression(self, agent, regression_df):
        """Test regression modeling."""
        result = await agent.execute(
            data=regression_df,
            target_column="target",
            task_type=TaskType.REGRESSION,
        )

        assert result.success
        assert result.data is not None
        assert len(result.data.models) == 1

        model = result.data.models[0]
        assert isinstance(model, ModelResult)
        assert model.task_type == TaskType.REGRESSION
        assert model.metrics.r2 is not None
        assert model.metrics.rmse is not None

    @pytest.mark.asyncio
    async def test_execute_regression_with_specific_model(self, agent, regression_df):
        """Test regression with specific model type."""
        result = await agent.execute(
            data=regression_df,
            target_column="target",
            model_type=ModelType.LINEAR_REGRESSION,
            task_type=TaskType.REGRESSION,
        )

        assert result.success
        model = result.data.models[0]
        assert model.model_type == ModelType.LINEAR_REGRESSION

    @pytest.mark.asyncio
    async def test_execute_classification(self, agent, classification_df):
        """Test classification modeling."""
        result = await agent.execute(
            data=classification_df,
            target_column="label",
            task_type=TaskType.CLASSIFICATION,
        )

        assert result.success
        assert result.data is not None
        assert len(result.data.models) == 1

        model = result.data.models[0]
        assert isinstance(model, ModelResult)
        assert model.task_type == TaskType.CLASSIFICATION
        assert model.metrics.accuracy is not None
        assert model.metrics.f1 is not None
        assert model.metrics.confusion_matrix is not None

    @pytest.mark.asyncio
    async def test_execute_classification_with_logistic(self, agent, classification_df):
        """Test classification with logistic regression."""
        result = await agent.execute(
            data=classification_df,
            target_column="label",
            model_type=ModelType.LOGISTIC_REGRESSION,
            task_type=TaskType.CLASSIFICATION,
        )

        assert result.success
        model = result.data.models[0]
        assert model.model_type == ModelType.LOGISTIC_REGRESSION

    @pytest.mark.asyncio
    async def test_execute_time_series_forecast(self, agent, time_series_df):
        """Test time series forecasting."""
        result = await agent.execute(
            data=time_series_df,
            target_column="value",
            date_column="date",
            task_type=TaskType.TIME_SERIES_FORECAST,
            periods=7,
        )

        assert result.success
        assert result.data is not None
        assert len(result.data.models) == 1

        model = result.data.models[0]
        assert isinstance(model, ForecastResult)
        assert len(model.predictions) == 7
        assert len(model.confidence_intervals) == 7
        assert model.prediction_dates is not None

    @pytest.mark.asyncio
    async def test_execute_with_no_data(self, agent):
        """Test error handling when no data provided."""
        result = await agent.execute(data=None, target_column="target")

        assert not result.success
        assert "no data" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_empty_data(self, agent):
        """Test error handling for empty DataFrame."""
        result = await agent.execute(
            data=pd.DataFrame(),
            target_column="target",
        )

        assert not result.success
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_without_target_column(self, agent, regression_df):
        """Test error handling when no target column specified."""
        result = await agent.execute(data=regression_df)

        assert not result.success
        assert "target_column" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_invalid_target_column(self, agent, regression_df):
        """Test error handling for invalid target column."""
        result = await agent.execute(
            data=regression_df,
            target_column="nonexistent",
        )

        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_infer_task_type_classification(self, agent, classification_df):
        """Test automatic task type inference for classification."""
        result = await agent.execute(
            data=classification_df,
            target_column="label",
            # No task_type specified - should infer classification
        )

        assert result.success
        model = result.data.models[0]
        assert model.task_type == TaskType.CLASSIFICATION

    @pytest.mark.asyncio
    async def test_infer_task_type_regression(self, agent, regression_df):
        """Test automatic task type inference for regression."""
        result = await agent.execute(
            data=regression_df,
            target_column="target",
            # No task_type specified - should infer regression
        )

        assert result.success
        model = result.data.models[0]
        assert model.task_type == TaskType.REGRESSION

    @pytest.mark.asyncio
    async def test_feature_importance_extracted(self, agent, regression_df):
        """Test that feature importance is extracted."""
        result = await agent.execute(
            data=regression_df,
            target_column="target",
            model_type=ModelType.RANDOM_FOREST_REGRESSOR,
        )

        assert result.success
        model = result.data.models[0]
        assert model.feature_importance is not None
        assert len(model.feature_importance.features) == 2
        assert len(model.feature_importance.importances) == 2

    @pytest.mark.asyncio
    async def test_cross_validation_performed(self, agent, regression_df):
        """Test that cross-validation is performed."""
        result = await agent.execute(
            data=regression_df,
            target_column="target",
            cv_folds=5,
        )

        assert result.success
        model = result.data.models[0]
        assert model.metrics.cv_scores is not None
        assert len(model.metrics.cv_scores) == 5
        assert model.metrics.cv_mean is not None
        assert model.metrics.cv_std is not None

    @pytest.mark.asyncio
    async def test_data_split_info(self, agent, regression_df):
        """Test that data split information is captured."""
        result = await agent.execute(
            data=regression_df,
            target_column="target",
            test_size=0.3,
        )

        assert result.success
        model = result.data.models[0]
        assert model.data_split.test_ratio == 0.3
        assert model.data_split.train_ratio == 0.7
        assert model.data_split.train_size > 0
        assert model.data_split.test_size > 0

    @pytest.mark.asyncio
    async def test_predictions_with_intervals(self, agent, regression_df):
        """Test that predictions include confidence intervals."""
        result = await agent.execute(
            data=regression_df,
            target_column="target",
            task_type=TaskType.REGRESSION,
        )

        assert result.success
        model = result.data.models[0]
        assert model.predictions.predictions is not None
        assert model.predictions.prediction_intervals is not None
        assert len(model.predictions.predictions) == len(model.predictions.prediction_intervals)

    @pytest.mark.asyncio
    async def test_diagnostics_computed(self, agent, regression_df):
        """Test that model diagnostics are computed for regression."""
        result = await agent.execute(
            data=regression_df,
            target_column="target",
            task_type=TaskType.REGRESSION,
        )

        assert result.success
        model = result.data.models[0]
        assert model.diagnostics.residual_mean is not None
        assert model.diagnostics.residual_std is not None
        assert model.diagnostics.durbin_watson is not None

    def test_get_model_options_regression(self, agent, regression_df):
        """Test getting model options for regression."""
        options = agent.get_model_options(
            regression_df,
            target_column="target",
            task_type=TaskType.REGRESSION,
        )

        assert len(options) >= 2
        assert any(opt.id == "random_forest_regressor" for opt in options)
        assert any(opt.recommended for opt in options)

    def test_get_model_options_classification(self, agent, classification_df):
        """Test getting model options for classification."""
        options = agent.get_model_options(
            classification_df,
            target_column="label",
            task_type=TaskType.CLASSIFICATION,
        )

        assert len(options) >= 2
        assert any(opt.id == "random_forest_classifier" for opt in options)
        assert any(opt.recommended for opt in options)

    def test_get_model_options_time_series(self, agent, time_series_df):
        """Test getting model options for time series."""
        options = agent.get_model_options(
            time_series_df,
            target_column="value",
            task_type=TaskType.TIME_SERIES_FORECAST,
        )

        assert len(options) >= 2
        assert any(opt.id == "exponential_smoothing" for opt in options)
        assert any(opt.recommended for opt in options)

    def test_format_output_regression(self, agent):
        """Test output formatting for regression."""
        output = ModelingOutput(
            models=[
                ModelResult(
                    model_type=ModelType.LINEAR_REGRESSION,
                    task_type=TaskType.REGRESSION,
                    data_split=DataSplit(
                        train_size=80,
                        test_size=20,
                        train_ratio=0.8,
                        test_ratio=0.2,
                        random_state=42,
                    ),
                    metrics=ModelMetrics(
                        model_type="linear_regression",
                        task_type="regression",
                        r2=0.85,
                        rmse=1.5,
                        mae=1.2,
                    ),
                    feature_importance=FeatureImportance(
                        features=["feature1", "feature2"],
                        importances=[0.6, 0.4],
                    ),
                    predictions=PredictionResult(
                        predictions=[1.0, 2.0, 3.0],
                        prediction_intervals=[(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)],
                    ),
                    diagnostics=ModelDiagnostics(
                        residual_mean=0.01,
                        residual_std=0.5,
                    ),
                    methodology="Test methodology",
                    limitations=["Test limitation"],
                    reproducibility={"random_state": 42},
                )
            ],
            summary="Test summary",
            recommendations=["Test recommendation"],
        )

        formatted = agent.format_output(output)

        assert "Linear Regression" in formatted
        assert "R-squared" in formatted
        assert "RMSE" in formatted
        assert "Feature Importance" in formatted
        assert "Test limitation" in formatted

    def test_format_output_classification(self, agent):
        """Test output formatting for classification."""
        output = ModelingOutput(
            models=[
                ModelResult(
                    model_type=ModelType.RANDOM_FOREST_CLASSIFIER,
                    task_type=TaskType.CLASSIFICATION,
                    data_split=DataSplit(
                        train_size=80,
                        test_size=20,
                        train_ratio=0.8,
                        test_ratio=0.2,
                        stratified=True,
                    ),
                    metrics=ModelMetrics(
                        model_type="random_forest_classifier",
                        task_type="classification",
                        accuracy=0.9,
                        precision=0.88,
                        recall=0.92,
                        f1=0.9,
                        confusion_matrix=[[40, 5], [3, 52]],
                    ),
                    feature_importance=FeatureImportance(
                        features=["feature1", "feature2"],
                        importances=[0.55, 0.45],
                    ),
                    predictions=PredictionResult(predictions=[0, 1, 1, 0]),
                    diagnostics=ModelDiagnostics(notes=["Test note"]),
                    methodology="Test methodology",
                    limitations=["Test limitation"],
                    reproducibility={"random_state": 42},
                )
            ],
            summary="Test summary",
        )

        formatted = agent.format_output(output)

        assert "Classification" in formatted
        assert "Accuracy" in formatted
        assert "F1 Score" in formatted
        assert "Stratified: True" in formatted

    def test_format_output_forecast(self, agent):
        """Test output formatting for time series forecast."""
        output = ModelingOutput(
            models=[
                ForecastResult(
                    model_type=ModelType.EXPONENTIAL_SMOOTHING,
                    periods=7,
                    predictions=[100, 101, 102, 103, 104, 105, 106],
                    prediction_dates=[
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-04",
                        "2024-01-05",
                        "2024-01-06",
                        "2024-01-07",
                    ],
                    confidence_intervals=[
                        (95, 105),
                        (93, 109),
                        (91, 113),
                        (89, 117),
                        (87, 121),
                        (85, 125),
                        (83, 129),
                    ],
                    confidence_level=0.95,
                    metrics=ModelMetrics(
                        model_type="exponential_smoothing",
                        task_type="time_series_forecast",
                        rmse=2.5,
                        mae=2.0,
                    ),
                    methodology="Test methodology",
                    limitations=["Test limitation"],
                    reproducibility={"periods": 7},
                )
            ],
            summary="Test summary",
        )

        formatted = agent.format_output(output)

        assert "Forecast" in formatted
        assert "Period" in formatted
        assert "Prediction" in formatted
        assert "95% CI" in formatted


class TestModelMetrics:
    """Test suite for ModelMetrics dataclass."""

    def test_to_dict_regression(self):
        """Test conversion to dictionary for regression metrics."""
        metrics = ModelMetrics(
            model_type="linear_regression",
            task_type="regression",
            rmse=1.5,
            mae=1.2,
            r2=0.85,
            mape=5.0,
        )

        d = metrics.to_dict()

        assert d["model_type"] == "linear_regression"
        assert d["task_type"] == "regression"
        assert d["rmse"] == 1.5
        assert d["r2"] == 0.85

    def test_to_dict_classification(self):
        """Test conversion to dictionary for classification metrics."""
        metrics = ModelMetrics(
            model_type="random_forest_classifier",
            task_type="classification",
            accuracy=0.9,
            precision=0.88,
            recall=0.92,
            f1=0.9,
            confusion_matrix=[[40, 5], [3, 52]],
        )

        d = metrics.to_dict()

        assert d["accuracy"] == 0.9
        assert d["f1"] == 0.9
        assert d["confusion_matrix"] == [[40, 5], [3, 52]]


class TestFeatureImportance:
    """Test suite for FeatureImportance dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary with ranking."""
        fi = FeatureImportance(
            features=["a", "b", "c"],
            importances=[0.2, 0.5, 0.3],
        )

        d = fi.to_dict()

        assert d["features"] == ["a", "b", "c"]
        assert d["importances"] == [0.2, 0.5, 0.3]

        # Ranking should be sorted by importance descending
        assert d["ranking"][0]["feature"] == "b"
        assert d["ranking"][0]["importance"] == 0.5


class TestModelingOutput:
    """Test suite for ModelingOutput dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        output = ModelingOutput(
            models=[],
            summary="Test summary",
            recommendations=["Rec 1"],
        )

        d = output.to_dict()

        assert d["models"] == []
        assert d["summary"] == "Test summary"
        assert "Rec 1" in d["recommendations"]
