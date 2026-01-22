"""
Modeling Agent for The Analyst platform.

Builds and evaluates predictive models for forecasting and classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.agents.base import AgentContext, AgentOption, AgentResult, BaseAgent
from src.prompts.agents import MODELING_PROMPT


class ModelType(str, Enum):
    """Types of predictive models available."""

    # Forecasting / Regression
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    LASSO_REGRESSION = "lasso_regression"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    GRADIENT_BOOSTING_REGRESSOR = "gradient_boosting_regressor"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    PROPHET = "prophet"
    ARIMA = "arima"

    # Classification
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    GRADIENT_BOOSTING_CLASSIFIER = "gradient_boosting_classifier"


class TaskType(str, Enum):
    """Types of modeling tasks."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    TIME_SERIES_FORECAST = "time_series_forecast"


@dataclass
class ModelMetrics:
    """Metrics for model evaluation."""

    # Common
    model_type: str
    task_type: str

    # Regression metrics
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None
    mape: float | None = None

    # Classification metrics
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    confusion_matrix: list[list[int]] | None = None

    # Cross-validation
    cv_scores: list[float] | None = None
    cv_mean: float | None = None
    cv_std: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_type": self.model_type,
            "task_type": self.task_type,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "mape": self.mape,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "confusion_matrix": self.confusion_matrix,
            "cv_scores": self.cv_scores,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
        }


@dataclass
class FeatureImportance:
    """Feature importance rankings."""

    features: list[str]
    importances: list[float]
    method: str = "model_native"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "features": self.features,
            "importances": self.importances,
            "method": self.method,
            "ranking": [
                {"feature": f, "importance": i}
                for f, i in sorted(
                    zip(self.features, self.importances),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ],
        }


@dataclass
class PredictionResult:
    """Result from model predictions."""

    predictions: list[float]
    prediction_intervals: list[tuple[float, float]] | None = None
    confidence_level: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predictions": self.predictions,
            "prediction_intervals": self.prediction_intervals,
            "confidence_level": self.confidence_level,
        }


@dataclass
class DataSplit:
    """Information about train/test split."""

    train_size: int
    test_size: int
    train_ratio: float
    test_ratio: float
    random_state: int | None = None
    stratified: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "train_size": self.train_size,
            "test_size": self.test_size,
            "train_ratio": self.train_ratio,
            "test_ratio": self.test_ratio,
            "random_state": self.random_state,
            "stratified": self.stratified,
        }


@dataclass
class ModelDiagnostics:
    """Model diagnostic information."""

    residual_mean: float | None = None
    residual_std: float | None = None
    residual_normality_p: float | None = None
    heteroscedasticity_test_p: float | None = None
    durbin_watson: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "residual_mean": self.residual_mean,
            "residual_std": self.residual_std,
            "residual_normality_p": self.residual_normality_p,
            "heteroscedasticity_test_p": self.heteroscedasticity_test_p,
            "durbin_watson": self.durbin_watson,
            "notes": self.notes,
        }


@dataclass
class ModelResult:
    """Complete result from model training and evaluation."""

    model_type: ModelType
    task_type: TaskType
    data_split: DataSplit
    metrics: ModelMetrics
    feature_importance: FeatureImportance | None
    predictions: PredictionResult
    diagnostics: ModelDiagnostics
    methodology: str
    limitations: list[str]
    reproducibility: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_type": self.model_type.value,
            "task_type": self.task_type.value,
            "data_split": self.data_split.to_dict(),
            "metrics": self.metrics.to_dict(),
            "feature_importance": (
                self.feature_importance.to_dict() if self.feature_importance else None
            ),
            "predictions": self.predictions.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "methodology": self.methodology,
            "limitations": self.limitations,
            "reproducibility": self.reproducibility,
        }


@dataclass
class ForecastResult:
    """Result from time series forecasting."""

    model_type: ModelType
    periods: int
    predictions: list[float]
    prediction_dates: list[str] | None
    confidence_intervals: list[tuple[float, float]]
    confidence_level: float
    metrics: ModelMetrics
    methodology: str
    limitations: list[str]
    reproducibility: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_type": self.model_type.value,
            "periods": self.periods,
            "predictions": self.predictions,
            "prediction_dates": self.prediction_dates,
            "confidence_intervals": self.confidence_intervals,
            "confidence_level": self.confidence_level,
            "metrics": self.metrics.to_dict(),
            "methodology": self.methodology,
            "limitations": self.limitations,
            "reproducibility": self.reproducibility,
        }


@dataclass
class ModelingOutput:
    """Complete output from the modeling agent."""

    models: list[ModelResult | ForecastResult] = field(default_factory=list)
    comparison: dict[str, Any] | None = None
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "models": [m.to_dict() for m in self.models],
            "comparison": self.comparison,
            "summary": self.summary,
            "recommendations": self.recommendations,
        }


class ModelingAgent(BaseAgent):
    """
    Agent responsible for building and evaluating predictive models.

    Single Job: Build and evaluate predictive models for forecasting and classification.
    """

    def __init__(self, context: AgentContext | None = None) -> None:
        """Initialize the modeling agent."""
        super().__init__(name="modeling", context=context)
        self._random_state = 42

    @property
    def system_prompt(self) -> str:
        """Get the agent's system prompt."""
        return MODELING_PROMPT

    async def execute(
        self,
        data: pd.DataFrame | None = None,
        target_column: str | None = None,
        feature_columns: list[str] | None = None,
        model_type: str | ModelType | None = None,
        task_type: str | TaskType | None = None,
        date_column: str | None = None,
        periods: int = 30,
        test_size: float = 0.2,
        cv_folds: int = 5,
        **kwargs: Any,
    ) -> AgentResult[ModelingOutput]:
        """
        Execute model building and evaluation.

        Args:
            data: DataFrame containing the data
            target_column: Column to predict
            feature_columns: Columns to use as features (None = all except target)
            model_type: Specific model type to use
            task_type: Type of task (regression, classification, time_series_forecast)
            date_column: Date column for time series forecasting
            periods: Number of periods to forecast (for time series)
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds

        Returns:
            AgentResult containing the modeling output
        """
        if data is None:
            return AgentResult.error_result("No data provided for modeling")

        if data.empty:
            return AgentResult.error_result("Data is empty")

        if target_column is None:
            return AgentResult.error_result("target_column is required for modeling")

        if target_column not in data.columns:
            return AgentResult.error_result(f"Target column '{target_column}' not found in data")

        # Convert string to enum if needed
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type.lower())
            except ValueError:
                model_type = None

        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type.lower())
            except ValueError:
                task_type = None

        # Infer task type if not provided
        if task_type is None:
            task_type = self._infer_task_type(data, target_column, date_column)

        self.log(f"Starting {task_type.value} modeling for target '{target_column}'")

        # Get feature columns
        if feature_columns is None:
            feature_columns = [c for c in data.columns if c != target_column and c != date_column]

        # Filter to numeric features only (for now)
        numeric_features = data[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_features and task_type != TaskType.TIME_SERIES_FORECAST:
            return AgentResult.error_result(
                f"No numeric features found. Available columns: {feature_columns}"
            )

        output = ModelingOutput()
        result: ModelResult | ForecastResult

        try:
            if task_type == TaskType.TIME_SERIES_FORECAST:
                result = self._run_time_series_forecast(
                    data, target_column, date_column, periods, model_type
                )
                output.models.append(result)
            elif task_type == TaskType.CLASSIFICATION:
                result = self._run_classification(
                    data, target_column, numeric_features, model_type, test_size, cv_folds
                )
                output.models.append(result)
            else:  # REGRESSION
                result = self._run_regression(
                    data, target_column, numeric_features, model_type, test_size, cv_folds
                )
                output.models.append(result)

            # Generate summary and recommendations
            output.summary = self._generate_summary(output.models)
            output.recommendations = self._generate_recommendations(output.models, task_type)

            self.log(f"Completed modeling with {len(output.models)} model(s)")

            return AgentResult.success_result(
                output,
                task_type=task_type.value,
                model_type=model_type.value if model_type else "auto",
                target=target_column,
            )

        except Exception as e:
            self.log(f"Modeling failed: {e}", level="ERROR")
            return AgentResult.error_result(f"Modeling failed: {e}")

    def _infer_task_type(
        self,
        data: pd.DataFrame,
        target_column: str,
        date_column: str | None,
    ) -> TaskType:
        """Infer the appropriate task type from the data."""
        # Check for time series
        if date_column and date_column in data.columns:
            return TaskType.TIME_SERIES_FORECAST

        # Check target column type
        target = data[target_column]
        if target.dtype == "object" or target.nunique() <= 10:
            return TaskType.CLASSIFICATION

        return TaskType.REGRESSION

    def _run_regression(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: list[str],
        model_type: ModelType | None,
        test_size: float,
        cv_folds: int,
    ) -> ModelResult:
        """Run regression modeling."""
        # Prepare data
        X = data[feature_columns].dropna()
        y = data.loc[X.index, target_column]

        # Remove any remaining NaN in target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self._random_state
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Select model
        if model_type is None:
            model_type = ModelType.RANDOM_FOREST_REGRESSOR

        model, methodology = self._get_regression_model(model_type)

        # Train model
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # MAPE (avoid division by zero)
        non_zero_mask = y_test != 0
        if non_zero_mask.sum() > 0:
            mape = (
                np.mean(
                    np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])
                )
                * 100
            )
        else:
            mape = None

        # Cross-validation
        cv_scores = cross_val_score(
            model, scaler.fit_transform(X), y, cv=cv_folds, scoring="neg_mean_squared_error"
        )
        cv_rmse = np.sqrt(-cv_scores)

        metrics = ModelMetrics(
            model_type=model_type.value,
            task_type=TaskType.REGRESSION.value,
            rmse=float(rmse),
            mae=float(mae),
            r2=float(r2),
            mape=float(mape) if mape is not None else None,
            cv_scores=[float(s) for s in cv_rmse],
            cv_mean=float(cv_rmse.mean()),
            cv_std=float(cv_rmse.std()),
        )

        # Feature importance
        feature_importance = self._get_feature_importance(model, feature_columns, model_type)

        # Prediction intervals (simple bootstrap-based approximation)
        residuals = y_test - y_pred
        residual_std = residuals.std()
        z_score = 1.96  # 95% confidence
        prediction_intervals = [
            (float(p - z_score * residual_std), float(p + z_score * residual_std)) for p in y_pred
        ]

        predictions = PredictionResult(
            predictions=[float(p) for p in y_pred],
            prediction_intervals=prediction_intervals,
            confidence_level=0.95,
        )

        # Diagnostics
        diagnostics = self._compute_regression_diagnostics(y_test, y_pred)

        # Data split info
        data_split = DataSplit(
            train_size=len(X_train),
            test_size=len(X_test),
            train_ratio=1 - test_size,
            test_ratio=test_size,
            random_state=self._random_state,
            stratified=False,
        )

        limitations = self._get_regression_limitations(data, metrics)

        return ModelResult(
            model_type=model_type,
            task_type=TaskType.REGRESSION,
            data_split=data_split,
            metrics=metrics,
            feature_importance=feature_importance,
            predictions=predictions,
            diagnostics=diagnostics,
            methodology=methodology,
            limitations=limitations,
            reproducibility={
                "random_state": self._random_state,
                "test_size": test_size,
                "cv_folds": cv_folds,
                "feature_columns": feature_columns,
                "target_column": target_column,
            },
        )

    def _run_classification(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: list[str],
        model_type: ModelType | None,
        test_size: float,
        cv_folds: int,
    ) -> ModelResult:
        """Run classification modeling."""
        # Prepare data
        X = data[feature_columns].dropna()
        y = data.loc[X.index, target_column]

        # Remove any remaining NaN in target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]

        # Encode target if needed
        label_encoder = None
        if y.dtype == "object":
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y), index=y.index)

        # Train/test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self._random_state, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Select model
        if model_type is None:
            model_type = ModelType.RANDOM_FOREST_CLASSIFIER

        model, methodology = self._get_classification_model(model_type)

        # Train model
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()

        # Cross-validation
        cv_scores = cross_val_score(
            model, scaler.fit_transform(X), y, cv=cv_folds, scoring="accuracy"
        )

        metrics = ModelMetrics(
            model_type=model_type.value,
            task_type=TaskType.CLASSIFICATION.value,
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            confusion_matrix=conf_matrix,
            cv_scores=[float(s) for s in cv_scores],
            cv_mean=float(cv_scores.mean()),
            cv_std=float(cv_scores.std()),
        )

        # Feature importance
        feature_importance = self._get_feature_importance(model, feature_columns, model_type)

        # Predictions (no intervals for classification)
        predictions = PredictionResult(
            predictions=[int(p) for p in y_pred],
            prediction_intervals=None,
            confidence_level=0.95,
        )

        # Diagnostics for classification
        diagnostics = ModelDiagnostics(
            notes=[
                f"Classes: {len(np.unique(y))}",
                f"Class distribution: {dict(pd.Series(y).value_counts())}",
            ]
        )

        # Data split info
        data_split = DataSplit(
            train_size=len(X_train),
            test_size=len(X_test),
            train_ratio=1 - test_size,
            test_ratio=test_size,
            random_state=self._random_state,
            stratified=True,
        )

        limitations = self._get_classification_limitations(data, metrics)

        return ModelResult(
            model_type=model_type,
            task_type=TaskType.CLASSIFICATION,
            data_split=data_split,
            metrics=metrics,
            feature_importance=feature_importance,
            predictions=predictions,
            diagnostics=diagnostics,
            methodology=methodology,
            limitations=limitations,
            reproducibility={
                "random_state": self._random_state,
                "test_size": test_size,
                "cv_folds": cv_folds,
                "feature_columns": feature_columns,
                "target_column": target_column,
                "label_encoding": label_encoder.classes_.tolist() if label_encoder else None,
            },
        )

    def _run_time_series_forecast(
        self,
        data: pd.DataFrame,
        target_column: str,
        date_column: str | None,
        periods: int,
        model_type: ModelType | None,
    ) -> ForecastResult:
        """Run time series forecasting."""
        # Prepare data
        df = data.copy()

        if date_column and date_column in df.columns:
            df = df.sort_values(date_column)
            dates = pd.to_datetime(df[date_column])
        else:
            dates = None

        y = np.asarray(df[target_column].dropna().values)

        if len(y) < 10:
            raise ValueError("Need at least 10 observations for time series forecasting")

        # Default to exponential smoothing
        if model_type is None:
            model_type = ModelType.EXPONENTIAL_SMOOTHING

        # Use simple exponential smoothing
        predictions, intervals, metrics, methodology = self._exponential_smoothing_forecast(
            y, periods
        )

        # Generate future dates if we have date column
        prediction_dates = None
        if dates is not None:
            last_date = dates.iloc[-1]
            freq = pd.infer_freq(dates)
            if freq:
                future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
                prediction_dates = [d.strftime("%Y-%m-%d") for d in future_dates]
            else:
                # Assume daily frequency
                future_dates = pd.date_range(start=last_date, periods=periods + 1, freq="D")[1:]
                prediction_dates = [d.strftime("%Y-%m-%d") for d in future_dates]

        limitations = [
            "Exponential smoothing assumes relatively stable patterns",
            "No seasonality adjustment in simple implementation",
            "Confidence intervals based on historical error variance",
            "Forecast accuracy decreases with longer horizons",
        ]

        return ForecastResult(
            model_type=model_type,
            periods=periods,
            predictions=predictions,
            prediction_dates=prediction_dates,
            confidence_intervals=intervals,
            confidence_level=0.95,
            metrics=metrics,
            methodology=methodology,
            limitations=limitations,
            reproducibility={
                "periods": periods,
                "target_column": target_column,
                "date_column": date_column,
                "n_observations": len(y),
            },
        )

    def _exponential_smoothing_forecast(
        self,
        y: np.ndarray,
        periods: int,
        alpha: float = 0.3,
    ) -> tuple[list[float], list[tuple[float, float]], ModelMetrics, str]:
        """Simple exponential smoothing forecast."""
        # Fit exponential smoothing
        n = len(y)
        smoothed = np.zeros(n)
        smoothed[0] = y[0]

        for i in range(1, n):
            smoothed[i] = alpha * y[i] + (1 - alpha) * smoothed[i - 1]

        # Forecast (flat forecast for simple exponential smoothing)
        last_smoothed = smoothed[-1]
        predictions = [float(last_smoothed) for _ in range(periods)]

        # Calculate error variance for confidence intervals
        residuals = y - smoothed
        residual_std = residuals.std()
        z_score = 1.96  # 95% confidence

        # Confidence intervals widen with forecast horizon
        intervals = []
        for h in range(1, periods + 1):
            width = z_score * residual_std * np.sqrt(h)
            intervals.append((float(last_smoothed - width), float(last_smoothed + width)))

        # Metrics on training data
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        mape = np.mean(np.abs(residuals / y)) * 100 if np.all(y != 0) else None

        metrics = ModelMetrics(
            model_type=ModelType.EXPONENTIAL_SMOOTHING.value,
            task_type=TaskType.TIME_SERIES_FORECAST.value,
            rmse=float(rmse),
            mae=float(mae),
            mape=float(mape) if mape is not None else None,
        )

        methodology = (
            f"Simple exponential smoothing with alpha={alpha}. "
            f"Confidence intervals based on residual variance with expanding horizon uncertainty."
        )

        return predictions, intervals, metrics, methodology

    def _get_regression_model(self, model_type: ModelType) -> tuple[Any, str]:
        """Get regression model instance and methodology description."""
        if model_type == ModelType.LINEAR_REGRESSION:
            return (
                LinearRegression(),
                "Ordinary Least Squares (OLS) linear regression minimizing sum of squared residuals.",
            )
        elif model_type == ModelType.RIDGE_REGRESSION:
            return (
                Ridge(alpha=1.0, random_state=self._random_state),
                "Ridge regression with L2 regularization to reduce overfitting.",
            )
        elif model_type == ModelType.LASSO_REGRESSION:
            return (
                Lasso(alpha=0.1, random_state=self._random_state),
                "Lasso regression with L1 regularization for feature selection.",
            )
        elif model_type == ModelType.RANDOM_FOREST_REGRESSOR:
            return (
                RandomForestRegressor(n_estimators=100, random_state=self._random_state),
                "Random Forest ensemble of 100 decision trees with bootstrap aggregating.",
            )
        elif model_type == ModelType.GRADIENT_BOOSTING_REGRESSOR:
            return (
                GradientBoostingRegressor(n_estimators=100, random_state=self._random_state),
                "Gradient Boosting ensemble with 100 sequential trees minimizing residual errors.",
            )
        else:
            # Default to Random Forest
            return (
                RandomForestRegressor(n_estimators=100, random_state=self._random_state),
                "Random Forest ensemble of 100 decision trees with bootstrap aggregating.",
            )

    def _get_classification_model(self, model_type: ModelType) -> tuple[Any, str]:
        """Get classification model instance and methodology description."""
        if model_type == ModelType.LOGISTIC_REGRESSION:
            return (
                LogisticRegression(random_state=self._random_state, max_iter=1000),
                "Logistic regression with L2 regularization for binary/multiclass classification.",
            )
        elif model_type == ModelType.RANDOM_FOREST_CLASSIFIER:
            return (
                RandomForestClassifier(n_estimators=100, random_state=self._random_state),
                "Random Forest ensemble of 100 decision trees with bootstrap aggregating for classification.",
            )
        elif model_type == ModelType.GRADIENT_BOOSTING_CLASSIFIER:
            return (
                GradientBoostingClassifier(n_estimators=100, random_state=self._random_state),
                "Gradient Boosting ensemble with 100 sequential trees for classification.",
            )
        else:
            # Default to Random Forest
            return (
                RandomForestClassifier(n_estimators=100, random_state=self._random_state),
                "Random Forest ensemble of 100 decision trees with bootstrap aggregating for classification.",
            )

    def _get_feature_importance(
        self,
        model: Any,
        feature_columns: list[str],
        model_type: ModelType,
    ) -> FeatureImportance | None:
        """Extract feature importance from model."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_.tolist()
            return FeatureImportance(
                features=feature_columns,
                importances=importances,
                method=(
                    "gini_importance" if "forest" in model_type.value.lower() else "model_native"
                ),
            )
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if len(coef.shape) > 1:
                coef = np.abs(coef).mean(axis=0)
            importances = np.abs(coef).tolist()
            return FeatureImportance(
                features=feature_columns,
                importances=importances,
                method="absolute_coefficients",
            )
        return None

    def _compute_regression_diagnostics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
    ) -> ModelDiagnostics:
        """Compute regression diagnostics."""
        residuals = y_true.values - y_pred

        # Basic residual statistics
        residual_mean = float(residuals.mean())
        residual_std = float(residuals.std())

        # Normality test (Shapiro-Wilk if small sample)
        from scipy import stats

        if len(residuals) <= 5000:
            _, normality_p = stats.shapiro(residuals[: min(5000, len(residuals))])
        else:
            normality_p = None

        # Durbin-Watson for autocorrelation
        diff_resid = np.diff(residuals)
        durbin_watson = float(np.sum(diff_resid**2) / np.sum(residuals**2))

        notes = []
        if abs(residual_mean) > 0.1 * residual_std:
            notes.append("Residuals show potential bias (mean significantly different from zero)")
        if normality_p is not None and normality_p < 0.05:
            notes.append("Residuals deviate from normality (Shapiro-Wilk p < 0.05)")
        if durbin_watson < 1.5 or durbin_watson > 2.5:
            notes.append(f"Potential autocorrelation in residuals (DW = {durbin_watson:.2f})")

        return ModelDiagnostics(
            residual_mean=residual_mean,
            residual_std=residual_std,
            residual_normality_p=float(normality_p) if normality_p is not None else None,
            durbin_watson=durbin_watson,
            notes=notes,
        )

    def _get_regression_limitations(
        self,
        data: pd.DataFrame,
        metrics: ModelMetrics,
    ) -> list[str]:
        """Get limitations for regression model."""
        limitations = []

        if len(data) < 100:
            limitations.append(f"Small sample size (n={len(data)}); model may not generalize well")

        if metrics.r2 is not None and metrics.r2 < 0.3:
            limitations.append(
                f"Low R-squared ({metrics.r2:.2f}) indicates limited predictive power"
            )

        if (
            metrics.cv_std is not None
            and metrics.cv_mean is not None
            and metrics.cv_std > 0.2 * metrics.cv_mean
        ):
            limitations.append("High cross-validation variance suggests unstable performance")

        limitations.append("Predictions assume similar data distribution as training data")
        limitations.append("Feature relationships assumed to remain constant")

        return limitations

    def _get_classification_limitations(
        self,
        data: pd.DataFrame,
        metrics: ModelMetrics,
    ) -> list[str]:
        """Get limitations for classification model."""
        limitations = []

        if len(data) < 100:
            limitations.append(f"Small sample size (n={len(data)}); model may not generalize well")

        if metrics.accuracy is not None and metrics.accuracy < 0.7:
            limitations.append(
                f"Moderate accuracy ({metrics.accuracy:.2f}) suggests room for improvement"
            )

        if metrics.cv_std is not None and metrics.cv_std > 0.1:
            limitations.append("Cross-validation variance indicates potential overfitting")

        limitations.append("Classification assumes similar class distributions in future data")
        limitations.append("Performance may vary for underrepresented classes")

        return limitations

    def _generate_summary(self, models: list[ModelResult | ForecastResult]) -> str:
        """Generate summary of modeling results."""
        if not models:
            return "No models were trained."

        parts = [f"Trained {len(models)} model(s)."]

        for model in models:
            if isinstance(model, ForecastResult):
                parts.append(
                    f"- {model.model_type.value}: Forecasted {model.periods} periods "
                    f"(RMSE: {model.metrics.rmse:.2f})"
                )
            else:
                if model.task_type == TaskType.CLASSIFICATION:
                    parts.append(
                        f"- {model.model_type.value}: Accuracy {model.metrics.accuracy:.2%}, "
                        f"F1 {model.metrics.f1:.2%}"
                    )
                else:
                    parts.append(
                        f"- {model.model_type.value}: R2 {model.metrics.r2:.2%}, "
                        f"RMSE {model.metrics.rmse:.2f}"
                    )

        return " ".join(parts)

    def _generate_recommendations(
        self,
        models: list[ModelResult | ForecastResult],
        task_type: TaskType,
    ) -> list[str]:
        """Generate recommendations based on model results."""
        recommendations = []

        for model in models:
            if isinstance(model, ForecastResult):
                if model.metrics.mape is not None and model.metrics.mape > 20:
                    recommendations.append(
                        "Consider more sophisticated time series models (Prophet, ARIMA) "
                        "for better accuracy"
                    )
                recommendations.append(
                    "Monitor forecast performance and retrain periodically with new data"
                )
            elif model.task_type == TaskType.CLASSIFICATION:
                if model.metrics.accuracy and model.metrics.accuracy < 0.8:
                    recommendations.append(
                        "Consider feature engineering or trying different model types"
                    )
                if model.feature_importance:
                    top_features = sorted(
                        zip(
                            model.feature_importance.features, model.feature_importance.importances
                        ),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:3]
                    recommendations.append(
                        f"Top predictive features: {', '.join(f[0] for f in top_features)}"
                    )
            else:  # Regression
                if model.metrics.r2 and model.metrics.r2 < 0.5:
                    recommendations.append(
                        "Low R-squared suggests missing important predictors; "
                        "consider additional features"
                    )
                if model.diagnostics.notes:
                    recommendations.extend(model.diagnostics.notes)

        if not recommendations:
            recommendations.append("Model performance is satisfactory for the given data")

        return recommendations

    def get_model_options(
        self,
        data: pd.DataFrame,
        target_column: str,
        task_type: TaskType | None = None,
    ) -> list[AgentOption]:
        """
        Get available model options based on the data and task.

        Args:
            data: DataFrame to model
            target_column: Target column to predict
            task_type: Type of modeling task

        Returns:
            List of model options for user selection
        """
        if task_type is None:
            task_type = self._infer_task_type(data, target_column, None)

        if task_type == TaskType.TIME_SERIES_FORECAST:
            return [
                AgentOption(
                    id="exponential_smoothing",
                    title="Exponential Smoothing",
                    description="Simple exponential smoothing for stable time series",
                    recommended=True,
                    pros=["Fast", "Robust", "Good for stable trends"],
                    cons=["No seasonality handling", "Simple patterns only"],
                    estimated_complexity="low",
                ),
                AgentOption(
                    id="prophet",
                    title="Prophet (Facebook)",
                    description="Handles seasonality and holidays automatically",
                    pros=["Seasonality detection", "Holiday effects", "Robust to outliers"],
                    cons=["Requires prophet library", "More computation"],
                    estimated_complexity="medium",
                ),
                AgentOption(
                    id="arima",
                    title="ARIMA",
                    description="Classical time series model with differencing",
                    pros=["Well-understood", "Good for short-term"],
                    cons=["Requires stationarity", "Parameter tuning needed"],
                    estimated_complexity="high",
                ),
            ]
        elif task_type == TaskType.CLASSIFICATION:
            return [
                AgentOption(
                    id="random_forest_classifier",
                    title="Random Forest",
                    description="Ensemble of decision trees with voting",
                    recommended=True,
                    pros=[
                        "Handles non-linear relationships",
                        "Feature importance",
                        "Low overfitting",
                    ],
                    cons=["Less interpretable", "Memory intensive"],
                    estimated_complexity="medium",
                ),
                AgentOption(
                    id="logistic_regression",
                    title="Logistic Regression",
                    description="Linear model for classification",
                    pros=["Interpretable coefficients", "Fast", "Probability outputs"],
                    cons=["Assumes linear boundaries", "May underfit complex data"],
                    estimated_complexity="low",
                ),
                AgentOption(
                    id="gradient_boosting_classifier",
                    title="Gradient Boosting",
                    description="Sequential ensemble with boosting",
                    pros=["High accuracy", "Handles imbalanced data"],
                    cons=["Prone to overfitting", "Slower training"],
                    estimated_complexity="high",
                ),
            ]
        else:  # Regression
            return [
                AgentOption(
                    id="random_forest_regressor",
                    title="Random Forest",
                    description="Ensemble of decision trees for regression",
                    recommended=True,
                    pros=["Handles non-linear relationships", "Feature importance", "Robust"],
                    cons=["Less interpretable", "Memory intensive"],
                    estimated_complexity="medium",
                ),
                AgentOption(
                    id="linear_regression",
                    title="Linear Regression",
                    description="Simple linear model (OLS)",
                    pros=["Highly interpretable", "Fast", "Statistical tests available"],
                    cons=["Assumes linear relationships", "Sensitive to outliers"],
                    estimated_complexity="low",
                ),
                AgentOption(
                    id="gradient_boosting_regressor",
                    title="Gradient Boosting",
                    description="Sequential ensemble with boosting",
                    pros=["High accuracy", "Handles complex patterns"],
                    cons=["Prone to overfitting", "Slower training"],
                    estimated_complexity="high",
                ),
            ]

    def format_output(self, output: ModelingOutput) -> str:
        """
        Format modeling output for display.

        Args:
            output: The modeling output to format

        Returns:
            Formatted markdown string
        """
        lines = ["# Predictive Modeling Results", ""]

        for model in output.models:
            if isinstance(model, ForecastResult):
                lines.extend(
                    [
                        f"## Time Series Forecast: {model.model_type.value.replace('_', ' ').title()}",
                        "",
                        "### Methodology",
                        model.methodology,
                        "",
                        f"### Forecast ({model.periods} periods)",
                        "",
                        "| Period | Prediction | 95% CI Lower | 95% CI Upper |",
                        "|--------|------------|--------------|--------------|",
                    ]
                )

                for i, (pred, (lower, upper)) in enumerate(
                    zip(model.predictions[:10], model.confidence_intervals[:10]), 1
                ):
                    date_str = (
                        model.prediction_dates[i - 1] if model.prediction_dates else f"Period {i}"
                    )
                    lines.append(f"| {date_str} | {pred:.2f} | {lower:.2f} | {upper:.2f} |")

                if len(model.predictions) > 10:
                    lines.append(
                        f"| ... | ({len(model.predictions) - 10} more periods) | ... | ... |"
                    )

                lines.extend(
                    [
                        "",
                        "### Performance Metrics",
                        f"- RMSE: {model.metrics.rmse:.4f}",
                        f"- MAE: {model.metrics.mae:.4f}",
                    ]
                )
                if model.metrics.mape is not None:
                    lines.append(f"- MAPE: {model.metrics.mape:.2f}%")

            else:  # ModelResult
                task_name = model.task_type.value.replace("_", " ").title()
                model_name = model.model_type.value.replace("_", " ").title()

                lines.extend(
                    [
                        f"## {task_name}: {model_name}",
                        "",
                        "### Methodology",
                        model.methodology,
                        "",
                        "### Data Split",
                        f"- Training: {model.data_split.train_size} samples ({model.data_split.train_ratio:.0%})",
                        f"- Test: {model.data_split.test_size} samples ({model.data_split.test_ratio:.0%})",
                        f"- Stratified: {model.data_split.stratified}",
                        "",
                        "### Performance Metrics",
                    ]
                )

                if model.task_type == TaskType.CLASSIFICATION:
                    lines.extend(
                        [
                            f"- Accuracy: {model.metrics.accuracy:.2%}",
                            f"- Precision: {model.metrics.precision:.2%}",
                            f"- Recall: {model.metrics.recall:.2%}",
                            f"- F1 Score: {model.metrics.f1:.2%}",
                        ]
                    )
                else:
                    lines.extend(
                        [
                            f"- R-squared: {model.metrics.r2:.4f}",
                            f"- RMSE: {model.metrics.rmse:.4f}",
                            f"- MAE: {model.metrics.mae:.4f}",
                        ]
                    )
                    if model.metrics.mape is not None:
                        lines.append(f"- MAPE: {model.metrics.mape:.2f}%")

                if model.metrics.cv_mean is not None:
                    lines.extend(
                        [
                            "",
                            f"### Cross-Validation ({len(model.metrics.cv_scores or [])} folds)",
                            f"- Mean: {model.metrics.cv_mean:.4f}",
                            f"- Std: {model.metrics.cv_std:.4f}",
                        ]
                    )

                if model.feature_importance:
                    lines.extend(
                        [
                            "",
                            "### Feature Importance",
                        ]
                    )
                    ranking = model.feature_importance.to_dict()["ranking"]
                    for i, item in enumerate(ranking[:10], 1):
                        lines.append(f"{i}. {item['feature']}: {item['importance']:.4f}")

                lines.extend(
                    [
                        "",
                        "### Limitations",
                    ]
                )
                for limitation in model.limitations:
                    lines.append(f"- {limitation}")

            lines.append("")

        if output.recommendations:
            lines.extend(
                [
                    "## Recommendations",
                ]
            )
            for rec in output.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        lines.extend(
            [
                "## Summary",
                output.summary,
            ]
        )

        return "\n".join(lines)
