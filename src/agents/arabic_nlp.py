"""
Arabic NLP Agent for The Analyst platform.

Processes Arabic text for sentiment analysis, NER, and dialect detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.agents.base import AgentContext, AgentResult, BaseAgent
from src.prompts.agents import ARABIC_NLP_PROMPT

# Optional heavy dependencies - graceful fallback if not installed
_TRANSFORMERS_AVAILABLE = False
_CAMEL_AVAILABLE = False

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    from camel_tools.dialectid import DIDModel6
    from camel_tools.ner import NERecognizer
    from camel_tools.sentiment import SentimentAnalyzer

    _CAMEL_AVAILABLE = True
except ImportError:
    pass


class ArabicDialect(str, Enum):
    """Arabic dialect categories."""

    MSA = "Modern Standard Arabic"
    GULF = "Gulf Arabic"
    EGYPTIAN = "Egyptian Arabic"
    LEVANTINE = "Levantine Arabic"
    MAGHREBI = "Maghrebi Arabic"
    MIXED = "Mixed/Unknown"


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class DialectResult:
    """Result from dialect detection."""

    primary_dialect: ArabicDialect
    confidence: float
    dialect_distribution: dict[str, float] = field(default_factory=dict)
    code_switching_detected: bool = False
    non_arabic_percentage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_dialect": self.primary_dialect.value,
            "confidence": self.confidence,
            "dialect_distribution": self.dialect_distribution,
            "code_switching_detected": self.code_switching_detected,
            "non_arabic_percentage": self.non_arabic_percentage,
        }


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""

    overall_sentiment: SentimentLabel
    confidence: float
    sentiment_distribution: dict[str, float] = field(default_factory=dict)
    sentence_sentiments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_sentiment": self.overall_sentiment.value,
            "confidence": self.confidence,
            "sentiment_distribution": self.sentiment_distribution,
            "sentence_sentiments": self.sentence_sentiments,
        }


@dataclass
class NamedEntity:
    """A named entity extracted from text."""

    text: str
    entity_type: str  # PERSON, ORG, LOC, etc.
    start_pos: int
    end_pos: int
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
        }


@dataclass
class NERResult:
    """Result from named entity recognition."""

    entities: list[NamedEntity] = field(default_factory=list)
    entity_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_counts": self.entity_counts,
        }


@dataclass
class ArabicNLPOutput:
    """Complete output from Arabic NLP analysis."""

    text_sample: str
    normalized_text: str
    dialect: DialectResult
    sentiment: SentimentResult
    ner: NERResult
    preprocessing_applied: list[str] = field(default_factory=list)
    model_info: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text_sample": self.text_sample,
            "normalized_text": self.normalized_text,
            "dialect": self.dialect.to_dict(),
            "sentiment": self.sentiment.to_dict(),
            "ner": self.ner.to_dict(),
            "preprocessing_applied": self.preprocessing_applied,
            "model_info": self.model_info,
        }


class ArabicNLPAgent(BaseAgent):
    """
    Agent responsible for Arabic NLP processing.

    Single Job: Process Arabic text for sentiment, NER, and dialect detection.
    """

    # Arabic character ranges
    ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")

    # Arabic normalization mappings
    ARABIC_NORMALIZATIONS = {
        "\u0622": "\u0627",  # Alef with madda -> Alef
        "\u0623": "\u0627",  # Alef with hamza above -> Alef
        "\u0625": "\u0627",  # Alef with hamza below -> Alef
        "\u0649": "\u064a",  # Alef maksura -> Yaa
        "\u0629": "\u0647",  # Taa marbuta -> Haa
    }

    def __init__(self, context: AgentContext | None = None) -> None:
        """Initialize the Arabic NLP agent."""
        super().__init__(name="arabic_nlp", context=context)

        # Track available models
        self._sentiment_pipeline = None
        self._ner_model = None
        self._dialect_model = None

        # Initialize models if available
        self._initialize_models()

    @property
    def system_prompt(self) -> str:
        """Get the agent's system prompt."""
        return ARABIC_NLP_PROMPT

    def _initialize_models(self) -> None:
        """Initialize NLP models if dependencies are available."""
        model_info = {}

        if _TRANSFORMERS_AVAILABLE:
            try:
                # Initialize sentiment analysis pipeline with MARBERT
                self._sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment",
                    tokenizer="CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment",
                )
                model_info["sentiment"] = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
                self.log("Sentiment pipeline initialized with CamelBERT")
            except Exception as e:
                self.log(f"Could not load sentiment model: {e}", level="WARNING")

        if _CAMEL_AVAILABLE:
            try:
                self._dialect_model = DIDModel6.pretrained()
                model_info["dialect"] = "CAMeL Tools DID-6"
                self.log("Dialect identification model initialized")
            except Exception as e:
                self.log(f"Could not load dialect model: {e}", level="WARNING")

            try:
                self._ner_model = NERecognizer.pretrained()
                model_info["ner"] = "CAMeL Tools NER"
                self.log("NER model initialized")
            except Exception as e:
                self.log(f"Could not load NER model: {e}", level="WARNING")

        self._model_info = model_info

    async def execute(
        self,
        text: str | list[str] | None = None,
        analyze_sentiment: bool = True,
        analyze_dialect: bool = True,
        extract_entities: bool = True,
        normalize_text: bool = True,
        **kwargs: Any,
    ) -> AgentResult[ArabicNLPOutput]:
        """
        Execute Arabic NLP analysis.

        Args:
            text: Text or list of texts to analyze
            analyze_sentiment: Whether to perform sentiment analysis
            analyze_dialect: Whether to detect dialect
            extract_entities: Whether to extract named entities
            normalize_text: Whether to normalize Arabic text

        Returns:
            AgentResult containing NLP analysis output
        """
        if text is None:
            return AgentResult.error_result("No text provided for analysis")

        # Handle list of texts
        if isinstance(text, list):
            text = " ".join(text)

        if not text.strip():
            return AgentResult.error_result("Empty text provided")

        self.log(f"Starting Arabic NLP analysis on {len(text)} characters")

        try:
            preprocessing_applied = []

            # Detect if text contains Arabic
            arabic_chars = len(self.ARABIC_PATTERN.findall(text))
            total_alpha = sum(1 for c in text if c.isalpha())
            arabic_percentage = (arabic_chars / total_alpha * 100) if total_alpha > 0 else 0

            if arabic_percentage < 10:
                return AgentResult.error_result(
                    f"Text appears to be primarily non-Arabic ({arabic_percentage:.1f}% Arabic characters). "
                    "Please provide Arabic text for analysis."
                )

            # Normalize text if requested
            normalized = text
            if normalize_text:
                normalized = self._normalize_arabic(text)
                preprocessing_applied.append("Arabic character normalization")

            # Sample for display
            text_sample = text[:200] + "..." if len(text) > 200 else text

            # Perform analyses
            dialect_result = (
                self._analyze_dialect(normalized) if analyze_dialect else self._default_dialect()
            )
            sentiment_result = (
                self._analyze_sentiment(normalized)
                if analyze_sentiment
                else self._default_sentiment()
            )
            ner_result = self._extract_entities(normalized) if extract_entities else NERResult()

            output = ArabicNLPOutput(
                text_sample=text_sample,
                normalized_text=normalized[:500] if len(normalized) > 500 else normalized,
                dialect=dialect_result,
                sentiment=sentiment_result,
                ner=ner_result,
                preprocessing_applied=preprocessing_applied,
                model_info=self._model_info,
            )

            self.log(
                f"Analysis complete: dialect={dialect_result.primary_dialect.value}, "
                f"sentiment={sentiment_result.overall_sentiment.value}, "
                f"entities={len(ner_result.entities)}"
            )

            return AgentResult.success_result(
                output,
                text_length=len(text),
                arabic_percentage=arabic_percentage,
            )

        except Exception as e:
            self.log(f"Arabic NLP analysis failed: {e}", level="ERROR")
            return AgentResult.error_result(f"Analysis failed: {e}")

    def _normalize_arabic(self, text: str) -> str:
        """
        Normalize Arabic text.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        # Apply character normalizations
        for original, normalized in self.ARABIC_NORMALIZATIONS.items():
            text = text.replace(original, normalized)

        # Remove diacritics (tashkeel)
        text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

        # Remove tatweel (kashida)
        text = text.replace("\u0640", "")

        # Normalize whitespace
        text = " ".join(text.split())

        return text

    def _analyze_dialect(self, text: str) -> DialectResult:
        """
        Detect Arabic dialect.

        Args:
            text: Text to analyze

        Returns:
            DialectResult with detected dialect
        """
        # Check for code-switching
        non_arabic = re.sub(self.ARABIC_PATTERN, "", text)
        non_arabic_alpha = sum(1 for c in non_arabic if c.isalpha())
        total_alpha = sum(1 for c in text if c.isalpha())

        code_switching = non_arabic_alpha > total_alpha * 0.2 if total_alpha > 0 else False
        non_arabic_pct = (non_arabic_alpha / total_alpha * 100) if total_alpha > 0 else 0

        if self._dialect_model and _CAMEL_AVAILABLE:
            try:
                # Use CAMeL Tools dialect identification
                predictions = self._dialect_model.predict([text])
                if predictions and len(predictions) > 0:
                    dialect_label = predictions[0]
                    dialect_map = {
                        "MSA": ArabicDialect.MSA,
                        "GLF": ArabicDialect.GULF,
                        "EGY": ArabicDialect.EGYPTIAN,
                        "LEV": ArabicDialect.LEVANTINE,
                        "NOR": ArabicDialect.MAGHREBI,
                        "LAV": ArabicDialect.LEVANTINE,
                        "MGR": ArabicDialect.MAGHREBI,
                    }
                    primary = dialect_map.get(dialect_label, ArabicDialect.MIXED)

                    return DialectResult(
                        primary_dialect=primary,
                        confidence=0.85,  # Model-based confidence
                        dialect_distribution={dialect_label: 0.85},
                        code_switching_detected=code_switching,
                        non_arabic_percentage=non_arabic_pct,
                    )
            except Exception as e:
                self.log(f"Dialect model error: {e}", level="WARNING")

        # Fallback: heuristic-based detection
        return self._heuristic_dialect_detection(text, code_switching, non_arabic_pct)

    def _heuristic_dialect_detection(
        self,
        text: str,
        code_switching: bool,
        non_arabic_pct: float,
    ) -> DialectResult:
        """
        Heuristic-based dialect detection fallback.

        Args:
            text: Text to analyze
            code_switching: Whether code-switching was detected
            non_arabic_pct: Percentage of non-Arabic characters

        Returns:
            DialectResult based on heuristics
        """
        # Simple heuristics based on common dialect markers
        text_lower = text.lower()

        # Egyptian markers
        egyptian_markers = ["ازيك", "ازاي", "كده", "ده", "دي", "اوي", "جدا", "ايه", "فين"]
        egyptian_count = sum(1 for m in egyptian_markers if m in text_lower)

        # Gulf markers
        gulf_markers = ["شلونك", "هلا", "شكو", "وين", "شنو", "زين", "الحين", "يالله"]
        gulf_count = sum(1 for m in gulf_markers if m in text_lower)

        # Levantine markers
        levantine_markers = ["كيفك", "شو", "هيك", "ليش", "هلأ", "منيح", "كتير"]
        levantine_count = sum(1 for m in levantine_markers if m in text_lower)

        # Maghrebi markers
        maghrebi_markers = ["واش", "كيفاش", "علاش", "بزاف", "زعما", "راني"]
        maghrebi_count = sum(1 for m in maghrebi_markers if m in text_lower)

        # Determine primary dialect
        counts = {
            ArabicDialect.EGYPTIAN: egyptian_count,
            ArabicDialect.GULF: gulf_count,
            ArabicDialect.LEVANTINE: levantine_count,
            ArabicDialect.MAGHREBI: maghrebi_count,
        }

        total_markers = sum(counts.values())

        if total_markers == 0:
            # Default to MSA if no dialect markers
            return DialectResult(
                primary_dialect=ArabicDialect.MSA,
                confidence=0.6,
                dialect_distribution={"MSA": 0.6, "Unknown": 0.4},
                code_switching_detected=code_switching,
                non_arabic_percentage=non_arabic_pct,
            )

        primary_dialect = max(counts, key=lambda d: counts.get(d, 0))
        confidence = counts[primary_dialect] / total_markers if total_markers > 0 else 0.5

        distribution = {d.value: c / total_markers for d, c in counts.items() if c > 0}

        return DialectResult(
            primary_dialect=primary_dialect,
            confidence=min(confidence, 0.7),  # Cap heuristic confidence
            dialect_distribution=distribution,
            code_switching_detected=code_switching,
            non_arabic_percentage=non_arabic_pct,
        )

    def _analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of Arabic text.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult with sentiment analysis
        """
        if self._sentiment_pipeline and _TRANSFORMERS_AVAILABLE:
            try:
                # Split into sentences for detailed analysis
                sentences = self._split_sentences(text)
                sentence_results = []
                sentiment_scores = {"positive": 0, "negative": 0, "neutral": 0}

                for sentence in sentences[:10]:  # Limit to first 10 sentences
                    if len(sentence.strip()) < 3:
                        continue

                    result = self._sentiment_pipeline(sentence[:512])[0]
                    label = result["label"].lower()
                    score = result["score"]

                    # Map labels
                    if "pos" in label:
                        sentiment_label = SentimentLabel.POSITIVE
                        sentiment_scores["positive"] += score
                    elif "neg" in label:
                        sentiment_label = SentimentLabel.NEGATIVE
                        sentiment_scores["negative"] += score
                    else:
                        sentiment_label = SentimentLabel.NEUTRAL
                        sentiment_scores["neutral"] += score

                    sentence_results.append(
                        {
                            "text": sentence[:100],
                            "sentiment": sentiment_label.value,
                            "confidence": score,
                        }
                    )

                # Calculate overall sentiment
                total = sum(sentiment_scores.values())
                if total > 0:
                    sentiment_scores = {k: v / total for k, v in sentiment_scores.items()}

                overall = max(sentiment_scores, key=sentiment_scores.get)
                overall_label = SentimentLabel(overall)

                return SentimentResult(
                    overall_sentiment=overall_label,
                    confidence=sentiment_scores[overall],
                    sentiment_distribution=sentiment_scores,
                    sentence_sentiments=sentence_results,
                )

            except Exception as e:
                self.log(f"Sentiment pipeline error: {e}", level="WARNING")

        # Fallback: lexicon-based sentiment
        return self._lexicon_sentiment(text)

    def _lexicon_sentiment(self, text: str) -> SentimentResult:
        """
        Lexicon-based sentiment analysis fallback.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult based on lexicon
        """
        # Simple Arabic sentiment lexicon
        positive_words = [
            "جميل",
            "ممتاز",
            "رائع",
            "جيد",
            "سعيد",
            "حب",
            "أحب",
            "شكرا",
            "مبارك",
            "نجاح",
            "فرح",
            "سرور",
            "تهانينا",
            "عظيم",
            "مذهل",
        ]
        negative_words = [
            "سيء",
            "فاشل",
            "كره",
            "أكره",
            "حزين",
            "مشكلة",
            "صعب",
            "مؤلم",
            "خطير",
            "سلبي",
            "فشل",
            "ضعيف",
            "رديء",
            "مزعج",
        ]

        text_lower = text.lower()

        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return SentimentResult(
                overall_sentiment=SentimentLabel.NEUTRAL,
                confidence=0.5,
                sentiment_distribution={"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                sentence_sentiments=[],
            )

        pos_ratio = pos_count / total
        neg_ratio = neg_count / total

        if pos_ratio > 0.6:
            overall = SentimentLabel.POSITIVE
            confidence = pos_ratio
        elif neg_ratio > 0.6:
            overall = SentimentLabel.NEGATIVE
            confidence = neg_ratio
        else:
            overall = SentimentLabel.NEUTRAL
            confidence = 0.5

        return SentimentResult(
            overall_sentiment=overall,
            confidence=min(confidence, 0.6),  # Cap lexicon confidence
            sentiment_distribution={
                "positive": pos_ratio,
                "negative": neg_ratio,
                "neutral": 1 - pos_ratio - neg_ratio if pos_ratio + neg_ratio < 1 else 0,
            },
            sentence_sentiments=[],
        )

    def _extract_entities(self, text: str) -> NERResult:
        """
        Extract named entities from Arabic text.

        Args:
            text: Text to analyze

        Returns:
            NERResult with extracted entities
        """
        entities: list[NamedEntity] = []
        entity_counts: dict[str, int] = {}

        if self._ner_model and _CAMEL_AVAILABLE:
            try:
                # Use CAMeL Tools NER
                ner_results = self._ner_model.predict_sentence(text.split())
                current_entity = None
                current_text = []
                current_start = 0

                for i, (word, tag) in enumerate(ner_results):
                    if tag.startswith("B-"):
                        # Save previous entity
                        if current_entity:
                            entity_text = " ".join(current_text)
                            entities.append(
                                NamedEntity(
                                    text=entity_text,
                                    entity_type=current_entity,
                                    start_pos=current_start,
                                    end_pos=current_start + len(entity_text),
                                )
                            )
                            entity_counts[current_entity] = entity_counts.get(current_entity, 0) + 1

                        # Start new entity
                        current_entity = tag[2:]
                        current_text = [word]
                        current_start = text.find(word)

                    elif tag.startswith("I-") and current_entity == tag[2:]:
                        current_text.append(word)

                    else:
                        # End of entity
                        if current_entity:
                            entity_text = " ".join(current_text)
                            entities.append(
                                NamedEntity(
                                    text=entity_text,
                                    entity_type=current_entity,
                                    start_pos=current_start,
                                    end_pos=current_start + len(entity_text),
                                )
                            )
                            entity_counts[current_entity] = entity_counts.get(current_entity, 0) + 1
                            current_entity = None
                            current_text = []

                # Handle last entity
                if current_entity:
                    entity_text = " ".join(current_text)
                    entities.append(
                        NamedEntity(
                            text=entity_text,
                            entity_type=current_entity,
                            start_pos=current_start,
                            end_pos=current_start + len(entity_text),
                        )
                    )
                    entity_counts[current_entity] = entity_counts.get(current_entity, 0) + 1

            except Exception as e:
                self.log(f"NER model error: {e}", level="WARNING")

        # Fallback: pattern-based entity extraction
        if not entities:
            entities, entity_counts = self._pattern_ner(text)

        return NERResult(entities=entities, entity_counts=entity_counts)

    def _pattern_ner(self, text: str) -> tuple[list[NamedEntity], dict[str, int]]:
        """
        Pattern-based NER fallback.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (entities list, entity counts dict)
        """
        entities: list[NamedEntity] = []
        entity_counts: dict[str, int] = {}

        # Simple patterns for common entities

        # Location patterns (common Arabic city/country names)
        locations = [
            "مصر",
            "السعودية",
            "الإمارات",
            "قطر",
            "الكويت",
            "البحرين",
            "عمان",
            "الأردن",
            "لبنان",
            "سوريا",
            "العراق",
            "فلسطين",
            "المغرب",
            "الجزائر",
            "تونس",
            "ليبيا",
            "السودان",
            "القاهرة",
            "الرياض",
            "دبي",
            "أبوظبي",
            "الدوحة",
            "بغداد",
        ]

        for loc in locations:
            if loc in text:
                start = text.find(loc)
                entities.append(
                    NamedEntity(
                        text=loc,
                        entity_type="LOC",
                        start_pos=start,
                        end_pos=start + len(loc),
                        confidence=0.7,
                    )
                )
                entity_counts["LOC"] = entity_counts.get("LOC", 0) + 1

        # Organization patterns (common Arabic org indicators)
        org_patterns = [
            r"شركة\s+[\u0600-\u06FF]+",
            r"مؤسسة\s+[\u0600-\u06FF]+",
            r"جامعة\s+[\u0600-\u06FF]+",
            r"وزارة\s+[\u0600-\u06FF]+",
        ]

        for pattern in org_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append(
                    NamedEntity(
                        text=match.group(),
                        entity_type="ORG",
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.6,
                    )
                )
                entity_counts["ORG"] = entity_counts.get("ORG", 0) + 1

        return entities, entity_counts

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split Arabic text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Arabic sentence delimiters
        delimiters = r"[.!?؟،\n]+"
        sentences = re.split(delimiters, text)
        return [s.strip() for s in sentences if s.strip()]

    def _default_dialect(self) -> DialectResult:
        """Return default dialect result when analysis is skipped."""
        return DialectResult(
            primary_dialect=ArabicDialect.MIXED,
            confidence=0.0,
            dialect_distribution={},
            code_switching_detected=False,
            non_arabic_percentage=0.0,
        )

    def _default_sentiment(self) -> SentimentResult:
        """Return default sentiment result when analysis is skipped."""
        return SentimentResult(
            overall_sentiment=SentimentLabel.NEUTRAL,
            confidence=0.0,
            sentiment_distribution={},
            sentence_sentiments=[],
        )

    def check_dependencies(self) -> dict[str, bool]:
        """
        Check which optional dependencies are available.

        Returns:
            Dictionary of dependency availability
        """
        return {
            "transformers": _TRANSFORMERS_AVAILABLE,
            "camel_tools": _CAMEL_AVAILABLE,
            "sentiment_model": self._sentiment_pipeline is not None,
            "dialect_model": self._dialect_model is not None,
            "ner_model": self._ner_model is not None,
        }

    def format_output(self, output: ArabicNLPOutput) -> str:
        """
        Format NLP output for display.

        Args:
            output: The NLP output to format

        Returns:
            Formatted markdown string
        """
        lines = [
            "# Arabic NLP Analysis",
            "",
            "## Text Sample",
            f"```\n{output.text_sample}\n```",
            "",
            "## Dialect Detection",
            f"- **Primary Dialect**: {output.dialect.primary_dialect.value}",
            f"- **Confidence**: {output.dialect.confidence:.1%}",
            f"- **Code-switching Detected**: {'Yes' if output.dialect.code_switching_detected else 'No'}",
        ]

        if output.dialect.dialect_distribution:
            lines.append("- **Dialect Distribution**:")
            for dialect, prob in output.dialect.dialect_distribution.items():
                lines.append(f"  - {dialect}: {prob:.1%}")

        lines.extend(
            [
                "",
                "## Sentiment Analysis",
                f"- **Overall Sentiment**: {output.sentiment.overall_sentiment.value.title()}",
                f"- **Confidence**: {output.sentiment.confidence:.1%}",
                "- **Sentiment Distribution**:",
            ]
        )

        for sentiment, prob in output.sentiment.sentiment_distribution.items():
            lines.append(f"  - {sentiment.title()}: {prob:.1%}")

        if output.ner.entities:
            lines.extend(
                [
                    "",
                    "## Named Entities",
                    "| Entity | Type | Count |",
                    "|--------|------|-------|",
                ]
            )

            # Aggregate by entity text and type
            entity_summary: dict[tuple[str, str], int] = {}
            for entity in output.ner.entities:
                key = (entity.text, entity.entity_type)
                entity_summary[key] = entity_summary.get(key, 0) + 1

            for (text, etype), count in sorted(entity_summary.items(), key=lambda x: -x[1]):
                lines.append(f"| {text} | {etype} | {count} |")

        if output.model_info:
            lines.extend(
                [
                    "",
                    "## Models Used",
                ]
            )
            for model_type, model_name in output.model_info.items():
                lines.append(f"- **{model_type.title()}**: {model_name}")

        return "\n".join(lines)
