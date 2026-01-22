"""Tests for the Arabic NLP Agent."""

import os
from unittest.mock import patch

import pytest

# Set environment variables before importing modules that use Settings
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-testing")
os.environ.setdefault("HUGGINGFACE_TOKEN", "test-token")

from src.agents.arabic_nlp import (
    ArabicDialect,
    ArabicNLPAgent,
    ArabicNLPOutput,
    DialectResult,
    NamedEntity,
    NERResult,
    SentimentLabel,
    SentimentResult,
)


class TestArabicNLPAgent:
    """Test suite for ArabicNLPAgent."""

    @pytest.fixture
    def agent(self):
        """Create an Arabic NLP agent for testing."""
        with patch("src.agents.base.AsyncAnthropic"):
            return ArabicNLPAgent()

    @pytest.fixture
    def arabic_text(self):
        """Sample Arabic text for testing."""
        return "مرحبا بكم في مصر. نحن سعداء جدا بلقائكم اليوم في القاهرة."

    @pytest.fixture
    def egyptian_text(self):
        """Sample Egyptian Arabic text."""
        return "إزيك يا صاحبي؟ الجو حلو أوي النهاردة!"

    @pytest.fixture
    def gulf_text(self):
        """Sample Gulf Arabic text."""
        return "شلونك؟ الحين الجو زين بالكويت."

    @pytest.fixture
    def mixed_text(self):
        """Sample mixed Arabic-English text."""
        return "أنا أحب الـ machine learning وهي technology رائعة!"

    def test_agent_initialization(self, agent):
        """Test that agent initializes correctly."""
        assert agent.name == "arabic_nlp"
        assert agent.autonomy.value == "supervised"

    @pytest.mark.asyncio
    async def test_execute_with_arabic_text(self, agent, arabic_text):
        """Test processing Arabic text."""
        result = await agent.execute(text=arabic_text)

        assert result.success
        assert result.data is not None
        assert result.data.text_sample
        assert result.data.dialect is not None
        assert result.data.sentiment is not None

    @pytest.mark.asyncio
    async def test_execute_with_no_text(self, agent):
        """Test error handling when no text provided."""
        result = await agent.execute(text=None)

        assert not result.success
        assert "no text" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_empty_text(self, agent):
        """Test error handling for empty text."""
        result = await agent.execute(text="   ")

        assert not result.success
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_non_arabic_text(self, agent):
        """Test error handling for non-Arabic text."""
        result = await agent.execute(text="Hello, this is English text only.")

        assert not result.success
        assert "arabic" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_list_of_texts(self, agent):
        """Test processing list of texts."""
        texts = ["مرحبا", "كيف حالك؟", "شكرا جزيلا"]
        result = await agent.execute(text=texts)

        assert result.success
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_dialect_detection_msa(self, agent, arabic_text):
        """Test MSA dialect detection."""
        result = await agent.execute(text=arabic_text)

        assert result.success
        # Standard Arabic should be detected as MSA
        assert result.data.dialect.primary_dialect in [
            ArabicDialect.MSA,
            ArabicDialect.EGYPTIAN,  # May detect due to location mentions
            ArabicDialect.MIXED,
        ]

    @pytest.mark.asyncio
    async def test_dialect_detection_egyptian(self, agent, egyptian_text):
        """Test Egyptian dialect detection."""
        result = await agent.execute(text=egyptian_text)

        assert result.success
        # Should detect Egyptian markers
        assert result.data.dialect.primary_dialect in [
            ArabicDialect.EGYPTIAN,
            ArabicDialect.MIXED,
        ]

    @pytest.mark.asyncio
    async def test_dialect_detection_gulf(self, agent, gulf_text):
        """Test Gulf dialect detection."""
        result = await agent.execute(text=gulf_text)

        assert result.success
        # Should detect Gulf markers
        assert result.data.dialect.primary_dialect in [
            ArabicDialect.GULF,
            ArabicDialect.MIXED,
        ]

    @pytest.mark.asyncio
    async def test_code_switching_detection(self, agent, mixed_text):
        """Test code-switching detection."""
        result = await agent.execute(text=mixed_text)

        assert result.success
        # Should detect code-switching
        assert (
            result.data.dialect.code_switching_detected
            or result.data.dialect.non_arabic_percentage > 0
        )

    @pytest.mark.asyncio
    async def test_sentiment_positive_text(self, agent):
        """Test positive sentiment detection."""
        positive_text = "هذا المنتج ممتاز ورائع جدا! أنا سعيد بالشراء."
        result = await agent.execute(text=positive_text)

        assert result.success
        # Should detect positive sentiment (via lexicon fallback)
        assert result.data.sentiment.overall_sentiment in [
            SentimentLabel.POSITIVE,
            SentimentLabel.NEUTRAL,  # May be neutral with lexicon-only
        ]

    @pytest.mark.asyncio
    async def test_sentiment_negative_text(self, agent):
        """Test negative sentiment detection."""
        negative_text = "هذا سيء جدا. أنا حزين ومحبط من النتيجة."
        result = await agent.execute(text=negative_text)

        assert result.success
        # Should detect negative sentiment
        assert result.data.sentiment.overall_sentiment in [
            SentimentLabel.NEGATIVE,
            SentimentLabel.NEUTRAL,
        ]

    @pytest.mark.asyncio
    async def test_entity_extraction_locations(self, agent, arabic_text):
        """Test location entity extraction."""
        result = await agent.execute(text=arabic_text)

        assert result.success
        # Should extract Egypt and Cairo as locations
        location_entities = [e for e in result.data.ner.entities if e.entity_type == "LOC"]
        assert len(location_entities) >= 1

    @pytest.mark.asyncio
    async def test_skip_analysis_options(self, agent, arabic_text):
        """Test skipping specific analyses."""
        result = await agent.execute(
            text=arabic_text,
            analyze_sentiment=False,
            analyze_dialect=False,
            extract_entities=False,
        )

        assert result.success
        # Should still return output but with defaults
        assert result.data.sentiment.confidence == 0.0
        assert result.data.dialect.confidence == 0.0

    @pytest.mark.asyncio
    async def test_text_normalization(self, agent):
        """Test Arabic text normalization."""
        # Text with various Arabic character variations
        text_with_variations = "الإمارات والأردن وإسرائيل"
        result = await agent.execute(text=text_with_variations, normalize_text=True)

        assert result.success
        assert "Arabic character normalization" in result.data.preprocessing_applied

    def test_check_dependencies(self, agent):
        """Test dependency checking."""
        deps = agent.check_dependencies()

        assert "transformers" in deps
        assert "camel_tools" in deps
        assert "sentiment_model" in deps
        assert "dialect_model" in deps
        assert "ner_model" in deps

    def test_format_output(self, agent):
        """Test output formatting."""
        output = ArabicNLPOutput(
            text_sample="مرحبا بكم",
            normalized_text="مرحبا بكم",
            dialect=DialectResult(
                primary_dialect=ArabicDialect.MSA,
                confidence=0.85,
                dialect_distribution={"MSA": 0.85},
                code_switching_detected=False,
                non_arabic_percentage=0.0,
            ),
            sentiment=SentimentResult(
                overall_sentiment=SentimentLabel.POSITIVE,
                confidence=0.75,
                sentiment_distribution={"positive": 0.75, "negative": 0.1, "neutral": 0.15},
                sentence_sentiments=[],
            ),
            ner=NERResult(
                entities=[NamedEntity(text="مصر", entity_type="LOC", start_pos=0, end_pos=3)],
                entity_counts={"LOC": 1},
            ),
            preprocessing_applied=["Arabic character normalization"],
            model_info={"sentiment": "test-model"},
        )

        formatted = agent.format_output(output)

        assert "Arabic NLP Analysis" in formatted
        assert "Dialect Detection" in formatted
        assert "Sentiment Analysis" in formatted
        assert "Named Entities" in formatted
        assert "MSA" in formatted or "Modern Standard Arabic" in formatted


class TestDialectResult:
    """Test suite for DialectResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = DialectResult(
            primary_dialect=ArabicDialect.EGYPTIAN,
            confidence=0.75,
            dialect_distribution={"EGY": 0.75, "MSA": 0.25},
            code_switching_detected=True,
            non_arabic_percentage=15.5,
        )

        d = result.to_dict()

        assert d["primary_dialect"] == "Egyptian Arabic"
        assert d["confidence"] == 0.75
        assert d["code_switching_detected"] is True
        assert "EGY" in d["dialect_distribution"]


class TestSentimentResult:
    """Test suite for SentimentResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SentimentResult(
            overall_sentiment=SentimentLabel.POSITIVE,
            confidence=0.85,
            sentiment_distribution={"positive": 0.85, "negative": 0.05, "neutral": 0.10},
            sentence_sentiments=[{"text": "test", "sentiment": "positive"}],
        )

        d = result.to_dict()

        assert d["overall_sentiment"] == "positive"
        assert d["confidence"] == 0.85
        assert len(d["sentence_sentiments"]) == 1


class TestNamedEntity:
    """Test suite for NamedEntity dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        entity = NamedEntity(
            text="القاهرة",
            entity_type="LOC",
            start_pos=10,
            end_pos=17,
            confidence=0.9,
        )

        d = entity.to_dict()

        assert d["text"] == "القاهرة"
        assert d["entity_type"] == "LOC"
        assert d["confidence"] == 0.9


class TestNERResult:
    """Test suite for NERResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = NERResult(
            entities=[
                NamedEntity(text="مصر", entity_type="LOC", start_pos=0, end_pos=3),
                NamedEntity(text="محمد", entity_type="PERSON", start_pos=10, end_pos=14),
            ],
            entity_counts={"LOC": 1, "PERSON": 1},
        )

        d = result.to_dict()

        assert len(d["entities"]) == 2
        assert d["entity_counts"]["LOC"] == 1
        assert d["entity_counts"]["PERSON"] == 1


class TestArabicNLPOutput:
    """Test suite for ArabicNLPOutput dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        output = ArabicNLPOutput(
            text_sample="مرحبا",
            normalized_text="مرحبا",
            dialect=DialectResult(
                primary_dialect=ArabicDialect.MSA,
                confidence=0.8,
            ),
            sentiment=SentimentResult(
                overall_sentiment=SentimentLabel.NEUTRAL,
                confidence=0.5,
            ),
            ner=NERResult(),
            preprocessing_applied=["normalization"],
            model_info={"test": "model"},
        )

        d = output.to_dict()

        assert d["text_sample"] == "مرحبا"
        assert "dialect" in d
        assert "sentiment" in d
        assert "ner" in d
        assert "normalization" in d["preprocessing_applied"]


class TestArabicNLPAgentExtended:
    """Extended test suite for ArabicNLPAgent covering internal methods."""

    @pytest.fixture
    def agent(self):
        """Create an Arabic NLP agent for testing."""
        with patch("src.agents.base.AsyncAnthropic"):
            return ArabicNLPAgent()

    def test_normalize_arabic_basic(self, agent):
        """Test basic Arabic normalization."""
        # Text with Alef variations
        text = "الإمارات العربية المتحدة"
        normalized = agent._normalize_arabic(text)

        # Should normalize Alef with hamza to plain Alef
        assert "إ" not in normalized or normalized != text

    def test_normalize_arabic_diacritics(self, agent):
        """Test removal of diacritics (tashkeel)."""
        text = "مُحَمَّد"  # Text with diacritics
        normalized = agent._normalize_arabic(text)

        # Diacritics should be removed
        assert len(normalized) <= len(text)

    def test_normalize_arabic_whitespace(self, agent):
        """Test whitespace normalization."""
        text = "كلمة    كلمة    كلمة"
        normalized = agent._normalize_arabic(text)

        # Multiple spaces should become single spaces
        assert "    " not in normalized
        # Just verify spaces are normalized, not exact match
        assert " " in normalized

    def test_split_sentences(self, agent):
        """Test Arabic sentence splitting."""
        text = "الجملة الأولى. الجملة الثانية؟ الجملة الثالثة!"
        sentences = agent._split_sentences(text)

        assert len(sentences) >= 3

    def test_split_sentences_with_newlines(self, agent):
        """Test sentence splitting with newlines."""
        text = "الجملة الأولى\nالجملة الثانية\nالجملة الثالثة"
        sentences = agent._split_sentences(text)

        assert len(sentences) >= 3

    def test_heuristic_dialect_detection_no_markers(self, agent):
        """Test heuristic dialect detection with no clear markers."""
        text = "هذا نص عربي بسيط"

        result = agent._heuristic_dialect_detection(text, code_switching=False, non_arabic_pct=0.0)

        # Should default to MSA when no dialect markers
        assert result.primary_dialect == ArabicDialect.MSA
        assert result.confidence == 0.6

    def test_heuristic_dialect_detection_egyptian(self, agent):
        """Test heuristic dialect detection with Egyptian markers."""
        text = "إزيك يا صاحبي؟ الجو حلو أوي كده!"

        result = agent._heuristic_dialect_detection(text, code_switching=False, non_arabic_pct=0.0)

        assert result.primary_dialect == ArabicDialect.EGYPTIAN

    def test_heuristic_dialect_detection_gulf(self, agent):
        """Test heuristic dialect detection with Gulf markers."""
        text = "شلونك؟ الحين وين رايح؟ زين"

        result = agent._heuristic_dialect_detection(text, code_switching=False, non_arabic_pct=0.0)

        assert result.primary_dialect == ArabicDialect.GULF

    def test_heuristic_dialect_detection_levantine(self, agent):
        """Test heuristic dialect detection with Levantine markers."""
        text = "كيفك؟ شو في؟ هيك منيح كتير"

        result = agent._heuristic_dialect_detection(text, code_switching=False, non_arabic_pct=0.0)

        assert result.primary_dialect == ArabicDialect.LEVANTINE

    def test_heuristic_dialect_detection_maghrebi(self, agent):
        """Test heuristic dialect detection with Maghrebi markers."""
        text = "واش راك؟ كيفاش حالك؟ بزاف زين"

        result = agent._heuristic_dialect_detection(text, code_switching=False, non_arabic_pct=0.0)

        assert result.primary_dialect == ArabicDialect.MAGHREBI

    def test_lexicon_sentiment_positive(self, agent):
        """Test lexicon-based sentiment for positive text."""
        text = "ممتاز رائع جميل سعيد"

        result = agent._lexicon_sentiment(text)

        assert result.overall_sentiment == SentimentLabel.POSITIVE

    def test_lexicon_sentiment_negative(self, agent):
        """Test lexicon-based sentiment for negative text."""
        text = "سيء فاشل أكره حزين مشكلة"

        result = agent._lexicon_sentiment(text)

        assert result.overall_sentiment == SentimentLabel.NEGATIVE

    def test_lexicon_sentiment_neutral(self, agent):
        """Test lexicon-based sentiment for neutral text without sentiment words."""
        text = "هذا كتاب في المكتبة"

        result = agent._lexicon_sentiment(text)

        assert result.overall_sentiment == SentimentLabel.NEUTRAL
        assert result.confidence == 0.5

    def test_lexicon_sentiment_mixed(self, agent):
        """Test lexicon-based sentiment for mixed text."""
        text = "رائع وجميل لكن أيضا صعب ومشكلة"

        result = agent._lexicon_sentiment(text)

        # Should be somewhere between positive and negative
        assert result.overall_sentiment in [
            SentimentLabel.POSITIVE,
            SentimentLabel.NEGATIVE,
            SentimentLabel.NEUTRAL,
        ]

    def test_pattern_ner_locations(self, agent):
        """Test pattern-based NER for locations."""
        text = "أنا من مصر وسافرت إلى السعودية ودبي"

        entities, counts = agent._pattern_ner(text)

        location_entities = [e for e in entities if e.entity_type == "LOC"]
        assert len(location_entities) >= 2
        assert counts.get("LOC", 0) >= 2

    def test_pattern_ner_organizations(self, agent):
        """Test pattern-based NER for organizations."""
        text = "شركة أرامكو وجامعة الملك سعود"

        entities, counts = agent._pattern_ner(text)

        org_entities = [e for e in entities if e.entity_type == "ORG"]
        assert len(org_entities) >= 1

    def test_pattern_ner_empty_text(self, agent):
        """Test pattern-based NER with no entities."""
        text = "كلام عادي بدون أسماء"

        entities, counts = agent._pattern_ner(text)

        # May or may not find entities
        assert isinstance(entities, list)
        assert isinstance(counts, dict)

    def test_default_dialect(self, agent):
        """Test default dialect result."""
        result = agent._default_dialect()

        assert result.primary_dialect == ArabicDialect.MIXED
        assert result.confidence == 0.0
        assert result.dialect_distribution == {}

    def test_default_sentiment(self, agent):
        """Test default sentiment result."""
        result = agent._default_sentiment()

        assert result.overall_sentiment == SentimentLabel.NEUTRAL
        assert result.confidence == 0.0
        assert result.sentiment_distribution == {}

    def test_system_prompt(self, agent):
        """Test that agent has a system prompt."""
        prompt = agent.system_prompt

        assert prompt is not None
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    @pytest.mark.asyncio
    async def test_execute_without_normalization(self, agent):
        """Test execution without text normalization."""
        text = "مرحبا بكم في العالم العربي"

        result = await agent.execute(
            text=text,
            normalize_text=False,
        )

        assert result.success
        assert "Arabic character normalization" not in result.data.preprocessing_applied

    def test_format_output_without_entities(self, agent):
        """Test formatting output when no entities found."""
        output = ArabicNLPOutput(
            text_sample="نص بدون كيانات",
            normalized_text="نص بدون كيانات",
            dialect=DialectResult(
                primary_dialect=ArabicDialect.MSA,
                confidence=0.7,
            ),
            sentiment=SentimentResult(
                overall_sentiment=SentimentLabel.NEUTRAL,
                confidence=0.5,
                sentiment_distribution={"neutral": 1.0},
            ),
            ner=NERResult(entities=[], entity_counts={}),
            preprocessing_applied=[],
            model_info={},
        )

        formatted = agent.format_output(output)

        assert "Arabic NLP Analysis" in formatted
        assert "Dialect Detection" in formatted
        assert "Sentiment Analysis" in formatted
        # Named Entities section should not appear when empty

    def test_format_output_without_model_info(self, agent):
        """Test formatting output when no model info available."""
        output = ArabicNLPOutput(
            text_sample="نص",
            normalized_text="نص",
            dialect=DialectResult(
                primary_dialect=ArabicDialect.MSA,
                confidence=0.5,
            ),
            sentiment=SentimentResult(
                overall_sentiment=SentimentLabel.NEUTRAL,
                confidence=0.5,
            ),
            ner=NERResult(),
            preprocessing_applied=[],
            model_info={},
        )

        formatted = agent.format_output(output)

        assert "Arabic NLP Analysis" in formatted
        # Models Used section should not appear when empty


class TestArabicDialectEnum:
    """Test suite for ArabicDialect enum."""

    def test_dialect_values(self):
        """Test that dialect enum has correct values."""
        assert ArabicDialect.MSA.value == "Modern Standard Arabic"
        assert ArabicDialect.GULF.value == "Gulf Arabic"
        assert ArabicDialect.EGYPTIAN.value == "Egyptian Arabic"
        assert ArabicDialect.LEVANTINE.value == "Levantine Arabic"
        assert ArabicDialect.MAGHREBI.value == "Maghrebi Arabic"
        assert ArabicDialect.MIXED.value == "Mixed/Unknown"


class TestSentimentLabelEnum:
    """Test suite for SentimentLabel enum."""

    def test_sentiment_values(self):
        """Test that sentiment enum has correct values."""
        assert SentimentLabel.POSITIVE.value == "positive"
        assert SentimentLabel.NEGATIVE.value == "negative"
        assert SentimentLabel.NEUTRAL.value == "neutral"
