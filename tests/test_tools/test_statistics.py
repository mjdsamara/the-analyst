"""Tests for the Statistics Toolkit."""

import numpy as np
import pandas as pd
import pytest

from src.tools.statistics import StatisticalResult, StatisticsToolkit


class TestStatisticsToolkit:
    """Test suite for StatisticsToolkit."""

    @pytest.fixture
    def toolkit(self):
        """Create a statistics toolkit for testing."""
        return StatisticsToolkit(confidence_level=0.95)

    @pytest.fixture
    def normal_data(self):
        """Create normally distributed data."""
        np.random.seed(42)
        return pd.Series(np.random.normal(100, 15, 100))

    @pytest.fixture
    def skewed_data(self):
        """Create skewed data."""
        np.random.seed(42)
        return pd.Series(np.random.exponential(10, 100))

    def test_descriptive_stats_mean(self, toolkit, normal_data):
        """Test descriptive statistics calculation."""
        results = toolkit.descriptive_stats(normal_data, name="test")

        assert "mean" in results
        assert results["mean"].confidence_interval is not None
        assert 95 < results["mean"].value < 105  # Should be near 100

    def test_descriptive_stats_includes_all_measures(self, toolkit, normal_data):
        """Test that all standard measures are included."""
        results = toolkit.descriptive_stats(normal_data)

        expected_measures = ["mean", "median", "std", "skewness", "kurtosis"]
        for measure in expected_measures:
            assert measure in results

    def test_normality_test_normal_data(self, toolkit, normal_data):
        """Test normality test with normal data."""
        result = toolkit.normality_test(normal_data)

        # Normal data should pass normality test (p > 0.05)
        assert result.p_value > 0.05
        assert "appears" in result.interpretation.lower()

    def test_normality_test_skewed_data(self, toolkit, skewed_data):
        """Test normality test with skewed data."""
        result = toolkit.normality_test(skewed_data)

        # Skewed data should fail normality test
        assert result.p_value < 0.05
        assert "does not appear" in result.interpretation.lower()

    def test_correlation_positive(self, toolkit):
        """Test correlation with positively correlated data."""
        x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = pd.Series([2, 4, 5, 4, 5, 7, 8, 9, 10, 10])

        result = toolkit.correlation(x, y)

        assert result.value > 0.8  # Strong positive correlation
        assert result.p_value < 0.05
        assert "positive" in result.interpretation.lower()

    def test_correlation_negative(self, toolkit):
        """Test correlation with negatively correlated data."""
        x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

        result = toolkit.correlation(x, y)

        assert result.value < -0.9  # Strong negative correlation
        assert "negative" in result.interpretation.lower()

    def test_correlation_methods(self, toolkit, normal_data):
        """Test different correlation methods."""
        x = normal_data
        y = normal_data + np.random.normal(0, 5, len(normal_data))

        for method in ["pearson", "spearman", "kendall"]:
            result = toolkit.correlation(x, y, method=method)
            assert result.value > 0.5  # Should show positive correlation

    def test_t_test_significant_difference(self, toolkit):
        """Test t-test with significantly different groups."""
        group1 = pd.Series([1, 2, 3, 4, 5])
        group2 = pd.Series([10, 11, 12, 13, 14])

        result = toolkit.t_test(group1, group2)

        assert result.p_value < 0.05
        assert "significant" in result.interpretation.lower()

    def test_t_test_no_significant_difference(self, toolkit):
        """Test t-test with similar groups."""
        np.random.seed(42)
        group1 = pd.Series(np.random.normal(100, 10, 30))
        group2 = pd.Series(np.random.normal(100, 10, 30))

        result = toolkit.t_test(group1, group2)

        assert result.p_value > 0.05
        assert "no significant" in result.interpretation.lower()

    def test_chi_square_test(self, toolkit):
        """Test chi-square test."""
        observed = pd.DataFrame(
            {
                "Yes": [50, 30],
                "No": [20, 40],
            },
            index=["Group A", "Group B"],
        )

        result = toolkit.chi_square_test(observed)

        assert result.p_value is not None
        assert result.effect_size is not None  # Cramér's V
        assert result.methodology == "Chi-square test of independence"

    def test_effect_size_interpretation(self, toolkit):
        """Test that effect sizes are interpreted."""
        group1 = pd.Series([1, 2, 3, 4, 5])
        group2 = pd.Series([10, 11, 12, 13, 14])

        result = toolkit.t_test(group1, group2)

        assert result.effect_size is not None
        # Large difference should have large effect size
        assert abs(result.effect_size) > 0.8

    def test_assumptions_included(self, toolkit, normal_data):
        """Test that assumptions are included in results."""
        result = toolkit.normality_test(normal_data)
        assert result.assumptions is not None
        assert len(result.assumptions) > 0


class TestStatisticalResult:
    """Test suite for StatisticalResult dataclass."""

    def test_result_creation(self):
        """Test creating a statistical result."""
        result = StatisticalResult(
            name="test",
            value=0.5,
            confidence_interval=(0.3, 0.7),
            p_value=0.01,
            effect_size=0.3,
            interpretation="Test interpretation",
            methodology="Test method",
        )

        assert result.name == "test"
        assert result.value == 0.5
        assert result.confidence_interval == (0.3, 0.7)

    def test_result_with_defaults(self):
        """Test result with default values."""
        result = StatisticalResult(name="test", value=1.0)

        assert result.confidence_interval is None
        assert result.p_value is None
        assert result.interpretation == ""


class TestStatisticsToolkitExtended:
    """Extended test suite for StatisticsToolkit covering additional cases."""

    @pytest.fixture
    def toolkit(self):
        """Create a statistics toolkit for testing."""
        return StatisticsToolkit(confidence_level=0.95)

    @pytest.fixture
    def toolkit_90(self):
        """Create a toolkit with 90% confidence level."""
        return StatisticsToolkit(confidence_level=0.90)

    # -------------------------------------------------------------------------
    # Normality Test Extended
    # -------------------------------------------------------------------------

    def test_normality_test_dagostino(self, toolkit):
        """Test D'Agostino-Pearson normality test."""
        np.random.seed(42)
        normal_data = pd.Series(np.random.normal(100, 15, 100))

        result = toolkit.normality_test(normal_data, test="dagostino")

        assert result.methodology == "D'Agostino-Pearson test"
        assert result.p_value is not None

    def test_normality_test_anderson(self, toolkit):
        """Test Anderson-Darling normality test."""
        np.random.seed(42)
        normal_data = pd.Series(np.random.normal(100, 15, 100))

        result = toolkit.normality_test(normal_data, test="anderson")

        assert result.methodology == "Anderson-Darling test"
        assert result.p_value is not None

    def test_normality_test_anderson_fails(self, toolkit):
        """Test Anderson-Darling test that fails normality."""
        np.random.seed(42)
        skewed_data = pd.Series(np.random.exponential(10, 100))

        result = toolkit.normality_test(skewed_data, test="anderson")

        assert result.methodology == "Anderson-Darling test"

    # -------------------------------------------------------------------------
    # T-Test Extended
    # -------------------------------------------------------------------------

    def test_t_test_one_sample(self, toolkit):
        """Test one-sample t-test."""
        data = pd.Series([5, 6, 5, 7, 6, 5, 8, 6, 5, 7])

        result = toolkit.t_test(data)  # Test against 0

        assert result.methodology == "One-sample t-test"
        assert result.p_value < 0.001  # Mean significantly different from 0

    def test_t_test_paired(self, toolkit):
        """Test paired samples t-test."""
        np.random.seed(42)
        before = pd.Series(np.random.normal(100, 10, 30))
        after = before + np.random.normal(5, 3, 30)

        result = toolkit.t_test(before, after, paired=True)

        assert "Paired" in result.methodology
        assert result.effect_size is not None

    def test_t_test_alternative_less(self, toolkit):
        """Test t-test with one-sided alternative (less)."""
        group1 = pd.Series([1, 2, 3, 4, 5])
        group2 = pd.Series([6, 7, 8, 9, 10])

        result = toolkit.t_test(group1, group2, alternative="less")

        assert result.p_value < 0.05

    def test_t_test_alternative_greater(self, toolkit):
        """Test t-test with one-sided alternative (greater)."""
        group1 = pd.Series([6, 7, 8, 9, 10])
        group2 = pd.Series([1, 2, 3, 4, 5])

        result = toolkit.t_test(group1, group2, alternative="greater")

        assert result.p_value < 0.05

    def test_t_test_effect_size_small(self, toolkit):
        """Test t-test with small effect size."""
        np.random.seed(42)
        group1 = pd.Series(np.random.normal(100, 15, 50))
        group2 = pd.Series(np.random.normal(101, 15, 50))

        result = toolkit.t_test(group1, group2)

        assert result.effect_size is not None
        assert abs(result.effect_size) < 0.5  # Small effect

    def test_t_test_effect_size_medium(self, toolkit):
        """Test t-test with medium effect size."""
        np.random.seed(42)
        group1 = pd.Series(np.random.normal(100, 15, 50))
        group2 = pd.Series(np.random.normal(107, 15, 50))

        result = toolkit.t_test(group1, group2)

        # Check that the interpretation contains effect size info
        assert "effect size" in result.interpretation.lower()

    # -------------------------------------------------------------------------
    # Chi-Square Extended
    # -------------------------------------------------------------------------

    def test_chi_square_goodness_of_fit(self, toolkit):
        """Test chi-square goodness of fit test with Series."""
        observed = pd.Series([20, 25, 30, 25])

        result = toolkit.chi_square_test(observed)

        assert "goodness of fit" in result.methodology.lower()
        assert result.p_value is not None

    def test_chi_square_goodness_of_fit_with_expected(self, toolkit):
        """Test chi-square with specified expected frequencies."""
        observed = pd.Series([40, 30, 20, 10])
        expected = pd.Series([25, 25, 25, 25])

        result = toolkit.chi_square_test(observed, expected)

        assert result.p_value < 0.05

    def test_chi_square_no_association(self, toolkit):
        """Test chi-square with no association."""
        observed = pd.DataFrame(
            {
                "Yes": [25, 25],
                "No": [25, 25],
            },
            index=["Male", "Female"],
        )

        result = toolkit.chi_square_test(observed)

        assert result.p_value > 0.05

    def test_chi_square_effect_size_small(self, toolkit):
        """Test chi-square with small effect size."""
        observed = pd.DataFrame(
            {
                "Yes": [26, 24],
                "No": [24, 26],
            },
            index=["Male", "Female"],
        )

        result = toolkit.chi_square_test(observed)

        assert result.effect_size is not None
        # Should be negligible or small effect

    def test_chi_square_effect_size_large(self, toolkit):
        """Test chi-square with large effect size."""
        observed = pd.DataFrame(
            {
                "Yes": [50, 5],
                "No": [5, 50],
            },
            index=["Male", "Female"],
        )

        result = toolkit.chi_square_test(observed)

        assert result.effect_size > 0.5

    # -------------------------------------------------------------------------
    # Correlation Extended
    # -------------------------------------------------------------------------

    def test_correlation_pearson_with_ci(self, toolkit):
        """Test Pearson correlation has confidence interval."""
        np.random.seed(42)
        x = pd.Series(np.random.normal(0, 1, 100))
        y = x + np.random.normal(0, 0.3, 100)

        result = toolkit.correlation(x, y, method="pearson")

        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2

    def test_correlation_spearman(self, toolkit):
        """Test Spearman rank correlation."""
        np.random.seed(42)
        x = pd.Series(np.random.normal(0, 1, 100))
        y = x**3

        result = toolkit.correlation(x, y, method="spearman")

        assert result.name == "spearman_correlation"
        # Spearman doesn't have CI
        assert result.confidence_interval is None

    def test_correlation_kendall(self, toolkit):
        """Test Kendall's tau correlation."""
        x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = pd.Series([1, 3, 2, 4, 6, 5, 7, 9, 8, 10])

        result = toolkit.correlation(x, y, method="kendall")

        assert result.name == "kendall_correlation"
        assert result.methodology == "Kendall's tau"

    def test_correlation_negligible(self, toolkit):
        """Test negligible correlation interpretation."""
        np.random.seed(123)
        x = pd.Series(np.random.normal(0, 1, 200))
        y = pd.Series(np.random.normal(0, 1, 200))

        result = toolkit.correlation(x, y)

        # Independent data should have negligible correlation
        assert abs(result.value) < 0.2

    def test_correlation_with_missing_values(self, toolkit):
        """Test correlation handles missing values."""
        x = pd.Series([1, 2, None, 4, 5])
        y = pd.Series([1, None, 3, 4, 5])

        result = toolkit.correlation(x, y)

        assert result.value is not None

    def test_correlation_effect_size_medium(self, toolkit):
        """Test medium correlation effect size interpretation."""
        np.random.seed(42)
        x = pd.Series(np.random.normal(0, 1, 100))
        y = x * 0.4 + np.random.normal(0, 0.9, 100)

        result = toolkit.correlation(x, y)

        # R-squared
        assert result.effect_size is not None
        assert result.effect_size == pytest.approx(result.value**2, rel=1e-6)

    # -------------------------------------------------------------------------
    # Interpretation Helper Tests
    # -------------------------------------------------------------------------

    def test_interpret_skewness_symmetric(self, toolkit):
        """Test skewness interpretation for symmetric data."""
        interpretation = toolkit._interpret_skewness(0.2)
        assert "symmetric" in interpretation.lower()

    def test_interpret_skewness_positive(self, toolkit):
        """Test skewness interpretation for positively skewed data."""
        interpretation = toolkit._interpret_skewness(1.5)
        assert "positively" in interpretation.lower() or "right" in interpretation.lower()

    def test_interpret_skewness_negative(self, toolkit):
        """Test skewness interpretation for negatively skewed data."""
        interpretation = toolkit._interpret_skewness(-1.5)
        assert "negatively" in interpretation.lower() or "left" in interpretation.lower()

    def test_interpret_kurtosis_mesokurtic(self, toolkit):
        """Test kurtosis interpretation for normal-like distribution."""
        interpretation = toolkit._interpret_kurtosis(0.2)
        assert "mesokurtic" in interpretation.lower() or "normal" in interpretation.lower()

    def test_interpret_kurtosis_leptokurtic(self, toolkit):
        """Test kurtosis interpretation for heavy-tailed distribution."""
        interpretation = toolkit._interpret_kurtosis(3.0)
        assert "leptokurtic" in interpretation.lower() or "heavy" in interpretation.lower()

    def test_interpret_kurtosis_platykurtic(self, toolkit):
        """Test kurtosis interpretation for light-tailed distribution."""
        interpretation = toolkit._interpret_kurtosis(-2.0)
        assert "platykurtic" in interpretation.lower() or "light" in interpretation.lower()

    # -------------------------------------------------------------------------
    # Confidence Level Tests
    # -------------------------------------------------------------------------

    def test_different_confidence_levels(self, toolkit, toolkit_90):
        """Test that confidence level affects intervals."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(100, 10, 50))

        result_95 = toolkit.descriptive_stats(data)
        result_90 = toolkit_90.descriptive_stats(data)

        ci_95 = result_95["mean"].confidence_interval
        ci_90 = result_90["mean"].confidence_interval

        # 95% CI should be wider than 90% CI
        width_95 = ci_95[1] - ci_95[0]
        width_90 = ci_90[1] - ci_90[0]
        assert width_95 > width_90

    def test_alpha_calculation(self, toolkit_90):
        """Test alpha is correctly calculated from confidence level."""
        assert toolkit_90.confidence_level == 0.90
        assert toolkit_90.alpha == pytest.approx(0.10)

    # -------------------------------------------------------------------------
    # Descriptive Stats Extended
    # -------------------------------------------------------------------------

    def test_descriptive_stats_with_named_series(self, toolkit):
        """Test descriptive stats with named Series."""
        data = pd.Series([1, 2, 3, 4, 5], name="my_column")
        results = toolkit.descriptive_stats(data)

        assert results["mean"].name == "my_column_mean"

    def test_descriptive_stats_with_custom_name(self, toolkit):
        """Test descriptive stats with custom name overrides series name."""
        data = pd.Series([1, 2, 3, 4, 5], name="series_name")
        results = toolkit.descriptive_stats(data, name="custom_name")

        assert results["mean"].name == "custom_name_mean"

    def test_descriptive_stats_with_missing_values(self, toolkit):
        """Test descriptive stats handles missing values."""
        data = pd.Series([1, 2, None, 4, 5, None])
        results = toolkit.descriptive_stats(data)

        # Mean of [1, 2, 4, 5] = 3.0
        assert results["mean"].value == pytest.approx(3.0)
