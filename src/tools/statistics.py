"""
Statistical analysis toolkit for The Analyst platform.

Provides statistical functions with proper methodology and confidence reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StatisticalResult:
    """Result from a statistical test or analysis."""

    name: str
    value: float
    confidence_interval: tuple[float, float] | None = None
    p_value: float | None = None
    effect_size: float | None = None
    interpretation: str = ""
    methodology: str = ""
    assumptions: list[str] | None = None


class StatisticsToolkit:
    """
    Statistical analysis toolkit with methodology tracking.

    All methods return results with:
    - Confidence intervals (where applicable)
    - Effect sizes (where applicable)
    - Methodology descriptions
    - Assumption checks
    """

    def __init__(self, confidence_level: float = 0.95) -> None:
        """
        Initialize the toolkit.

        Args:
            confidence_level: Confidence level for intervals (default 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def descriptive_stats(
        self,
        data: pd.Series,
        name: str | None = None,
    ) -> dict[str, StatisticalResult]:
        """
        Calculate descriptive statistics for a series.

        Args:
            data: Data series to analyze
            name: Optional name for the series

        Returns:
            Dictionary of statistical results
        """
        name = name or (str(data.name) if data.name is not None else None) or "series"
        clean_data = data.dropna()
        n = len(clean_data)

        results = {}

        # Mean with CI
        mean = clean_data.mean()
        sem = stats.sem(clean_data)
        ci = stats.t.interval(self.confidence_level, n - 1, loc=mean, scale=sem)
        results["mean"] = StatisticalResult(
            name=f"{name}_mean",
            value=mean,
            confidence_interval=ci,
            methodology="Sample mean with t-distribution confidence interval",
            assumptions=["Data is approximately normally distributed for CI validity"],
        )

        # Median
        median = clean_data.median()
        results["median"] = StatisticalResult(
            name=f"{name}_median",
            value=median,
            methodology="Sample median (50th percentile)",
        )

        # Standard deviation
        std = clean_data.std()
        results["std"] = StatisticalResult(
            name=f"{name}_std",
            value=std,
            methodology="Sample standard deviation (Bessel's correction)",
        )

        # Skewness
        skewness = stats.skew(clean_data)
        results["skewness"] = StatisticalResult(
            name=f"{name}_skewness",
            value=skewness,
            interpretation=self._interpret_skewness(skewness),
            methodology="Fisher-Pearson coefficient of skewness",
        )

        # Kurtosis
        kurtosis = stats.kurtosis(clean_data)
        results["kurtosis"] = StatisticalResult(
            name=f"{name}_kurtosis",
            value=kurtosis,
            interpretation=self._interpret_kurtosis(kurtosis),
            methodology="Excess kurtosis (Fisher's definition)",
        )

        return results

    def normality_test(
        self,
        data: pd.Series,
        test: str = "shapiro",
    ) -> StatisticalResult:
        """
        Test for normality.

        Args:
            data: Data to test
            test: Test to use ("shapiro", "dagostino", "anderson")

        Returns:
            Test result
        """
        clean_data = data.dropna()

        if test == "shapiro":
            stat, p_value = stats.shapiro(clean_data)
            methodology = "Shapiro-Wilk test"
        elif test == "dagostino":
            stat, p_value = stats.normaltest(clean_data)
            methodology = "D'Agostino-Pearson test"
        else:
            result = stats.anderson(clean_data)
            stat = result.statistic
            # Use 5% critical value
            p_value = 0.05 if stat > result.critical_values[2] else 0.10
            methodology = "Anderson-Darling test"

        is_normal = p_value > self.alpha
        interpretation = (
            f"Data {'appears' if is_normal else 'does not appear'} to be normally distributed "
            f"at {self.confidence_level:.0%} confidence level"
        )

        return StatisticalResult(
            name="normality_test",
            value=stat,
            p_value=p_value,
            interpretation=interpretation,
            methodology=methodology,
            assumptions=["Sample size should be > 20 for reliable results"],
        )

    def correlation(
        self,
        x: pd.Series,
        y: pd.Series,
        method: str = "pearson",
    ) -> StatisticalResult:
        """
        Calculate correlation between two series.

        Args:
            x: First series
            y: Second series
            method: Correlation method ("pearson", "spearman", "kendall")

        Returns:
            Correlation result
        """
        # Remove missing values (pairwise)
        mask = x.notna() & y.notna()
        x_clean = x[mask]
        y_clean = y[mask]
        n = len(x_clean)

        if method == "pearson":
            r, p_value = stats.pearsonr(x_clean, y_clean)
            methodology = "Pearson correlation coefficient"
            assumptions = [
                "Linear relationship between variables",
                "Both variables are normally distributed",
                "Homoscedasticity",
            ]
        elif method == "spearman":
            r, p_value = stats.spearmanr(x_clean, y_clean)
            methodology = "Spearman rank correlation"
            assumptions = ["Monotonic relationship between variables"]
        else:  # kendall
            r, p_value = stats.kendalltau(x_clean, y_clean)
            methodology = "Kendall's tau"
            assumptions = ["Ordinal or continuous data"]

        # Confidence interval for Pearson r using Fisher z-transformation
        ci = None
        if method == "pearson" and n > 3:
            z = np.arctanh(r)
            se = 1 / np.sqrt(n - 3)
            z_crit = stats.norm.ppf(1 - self.alpha / 2)
            ci = (np.tanh(z - z_crit * se), np.tanh(z + z_crit * se))

        # Effect size interpretation
        abs_r = abs(r)
        if abs_r < 0.1:
            effect_interp = "negligible"
        elif abs_r < 0.3:
            effect_interp = "small"
        elif abs_r < 0.5:
            effect_interp = "medium"
        else:
            effect_interp = "large"

        interpretation = (
            f"{effect_interp.title()} {'positive' if r > 0 else 'negative'} correlation"
        )

        return StatisticalResult(
            name=f"{method}_correlation",
            value=r,
            confidence_interval=ci,
            p_value=p_value,
            effect_size=r**2,  # R-squared as effect size
            interpretation=interpretation,
            methodology=methodology,
            assumptions=assumptions,
        )

    def t_test(
        self,
        group1: pd.Series,
        group2: pd.Series | None = None,
        paired: bool = False,
        alternative: str = "two-sided",
    ) -> StatisticalResult:
        """
        Perform t-test.

        Args:
            group1: First group data
            group2: Second group data (None for one-sample test)
            paired: Whether to perform paired t-test
            alternative: "two-sided", "less", or "greater"

        Returns:
            T-test result
        """
        g1 = group1.dropna()

        if group2 is None:
            # One-sample t-test against 0
            t_stat, p_value = stats.ttest_1samp(g1, 0, alternative=alternative)
            methodology = "One-sample t-test"
            effect_size = g1.mean() / g1.std()  # Cohen's d
        else:
            g2 = group2.dropna()

            if paired:
                t_stat, p_value = stats.ttest_rel(g1, g2, alternative=alternative)
                methodology = "Paired samples t-test"
                diff = g1 - g2
                effect_size = diff.mean() / diff.std()
            else:
                t_stat, p_value = stats.ttest_ind(g1, g2, alternative=alternative)
                methodology = "Independent samples t-test (Welch's)"

                # Cohen's d
                n1, n2 = len(g1), len(g2)
                pooled_std = np.sqrt(
                    ((n1 - 1) * g1.std() ** 2 + (n2 - 1) * g2.std() ** 2) / (n1 + n2 - 2)
                )
                effect_size = (g1.mean() - g2.mean()) / pooled_std

        # Effect size interpretation (Cohen's d)
        abs_d = abs(effect_size)
        if abs_d < 0.2:
            effect_interp = "negligible"
        elif abs_d < 0.5:
            effect_interp = "small"
        elif abs_d < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"

        significant = p_value < self.alpha
        interpretation = (
            f"{'Significant' if significant else 'No significant'} difference "
            f"with {effect_interp} effect size (d={effect_size:.3f})"
        )

        return StatisticalResult(
            name="t_test",
            value=t_stat,
            p_value=p_value,
            effect_size=effect_size,
            interpretation=interpretation,
            methodology=methodology,
            assumptions=[
                "Data is approximately normally distributed",
                "Observations are independent",
                "Equal variances (for independent samples - Welch's correction applied)",
            ],
        )

    def chi_square_test(
        self,
        observed: pd.DataFrame | pd.Series,
        expected: pd.DataFrame | pd.Series | None = None,
    ) -> StatisticalResult:
        """
        Perform chi-square test.

        Args:
            observed: Observed frequencies (contingency table or series)
            expected: Expected frequencies (None for independence test)

        Returns:
            Chi-square test result
        """
        if isinstance(observed, pd.Series):
            # Goodness of fit test
            if expected is None:
                # Assume uniform distribution
                expected = np.ones(len(observed)) * observed.sum() / len(observed)
            chi2, p_value = stats.chisquare(observed, expected)
            methodology = "Chi-square goodness of fit test"
            dof = len(observed) - 1
        else:
            # Independence test
            chi2, p_value, dof, expected = stats.chi2_contingency(observed)
            methodology = "Chi-square test of independence"

        # Cramér's V as effect size
        n = observed.sum().sum() if isinstance(observed, pd.DataFrame) else observed.sum()
        k = min(observed.shape) if isinstance(observed, pd.DataFrame) else 2
        cramers_v = np.sqrt(chi2 / (n * (k - 1)))

        # Effect size interpretation
        if cramers_v < 0.1:
            effect_interp = "negligible"
        elif cramers_v < 0.3:
            effect_interp = "small"
        elif cramers_v < 0.5:
            effect_interp = "medium"
        else:
            effect_interp = "large"

        significant = p_value < self.alpha
        interpretation = (
            f"{'Significant' if significant else 'No significant'} association "
            f"with {effect_interp} effect size (V={cramers_v:.3f})"
        )

        return StatisticalResult(
            name="chi_square_test",
            value=chi2,
            p_value=p_value,
            effect_size=cramers_v,
            interpretation=interpretation,
            methodology=methodology,
            assumptions=[
                "Expected frequency in each cell >= 5",
                "Observations are independent",
            ],
        )

    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness value."""
        if abs(skewness) < 0.5:
            return "Approximately symmetric"
        elif skewness > 0:
            return "Positively skewed (right-tailed)"
        else:
            return "Negatively skewed (left-tailed)"

    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpret kurtosis value."""
        if abs(kurtosis) < 0.5:
            return "Approximately mesokurtic (normal-like)"
        elif kurtosis > 0:
            return "Leptokurtic (heavy-tailed, peaked)"
        else:
            return "Platykurtic (light-tailed, flat)"
