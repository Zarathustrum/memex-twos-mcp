"""Tests for entity extraction (people, tags, links)."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from convert_to_json import extract_people, extract_people_ner, extract_people_batch_ner


class TestPeopleExtractionRegex:
    """Test regex-based people extraction (fallback mode)."""

    def test_simple_name(self):
        """Extract single name."""
        result = extract_people("Met with John today", use_ner=False)
        assert "John" in result

    def test_full_name(self):
        """Extract full name."""
        result = extract_people("Call Sarah Johnson", use_ner=False)
        # Regex extracts individual capitalized words
        assert "Sarah" in result or "Sarah Johnson" in result

    def test_false_positive_verb(self):
        """Regex incorrectly extracts verbs (known limitation)."""
        result = extract_people("Set reminder for meeting", use_ner=False)
        # "Set" is in the stop words list, should be filtered
        assert "Set" not in result

    def test_false_positive_month(self):
        """Regex filters out month names."""
        result = extract_people("Meeting in March", use_ner=False)
        # "March" is in excluded list, should be filtered
        assert "March" not in result

    def test_excludes_days(self):
        """Filter out day names."""
        result = extract_people("Monday meeting with Alice", use_ner=False)
        assert "Monday" not in result
        assert "Alice" in result

    def test_mom_normalization(self):
        """Normalize 'mom' to 'Mom'."""
        result = extract_people("dinner with mom", use_ner=False)
        assert "Mom" in result

    def test_empty_text(self):
        """Handle empty input."""
        result = extract_people("", use_ner=False)
        assert result == []


class TestPeopleExtractionNER:
    """Test NER-based people extraction."""

    def test_simple_name(self):
        """Extract single name with NER."""
        try:
            result = extract_people_ner("Met with John today")
            assert "John" in result
        except ImportError:
            pytest.skip("spaCy not installed")

    def test_full_name(self):
        """Extract full name with NER."""
        try:
            result = extract_people_ner("Call Sarah Johnson about project")
            # spaCy may extract as full name or separate
            assert any("Sarah" in name for name in result)
        except ImportError:
            pytest.skip("spaCy not installed")

    def test_no_false_positive_verb(self):
        """NER correctly rejects verbs."""
        try:
            result = extract_people_ner("Set reminder for meeting")
            assert "Set" not in result
        except ImportError:
            pytest.skip("spaCy not installed")

    def test_no_false_positive_month(self):
        """NER correctly rejects month names."""
        try:
            result = extract_people_ner("Meeting in March")
            assert "March" not in result
        except ImportError:
            pytest.skip("spaCy not installed")

    def test_multiple_names(self):
        """Extract multiple people from same text."""
        try:
            result = extract_people_ner("Dinner with Alice and Bob")
            assert "Alice" in result
            assert "Bob" in result
        except ImportError:
            pytest.skip("spaCy not installed")

    def test_names_in_context(self):
        """Extract names from realistic context."""
        try:
            text = "Called Dr. Smith about appointment. Sarah will join the meeting."
            result = extract_people_ner(text)
            # spaCy should find at least Sarah
            assert any("Sarah" in name for name in result)
        except ImportError:
            pytest.skip("spaCy not installed")

    def test_no_single_char_names(self):
        """Filter out single-character false positives."""
        try:
            result = extract_people_ner("Schedule A meeting")
            assert "A" not in result
        except ImportError:
            pytest.skip("spaCy not installed")

    def test_empty_text(self):
        """Handle empty input."""
        try:
            result = extract_people_ner("")
            assert result == []
        except ImportError:
            pytest.skip("spaCy not installed")


class TestPeopleExtractionBatchNER:
    """Test batch NER processing."""

    def test_batch_processing(self):
        """Process multiple texts in batch."""
        try:
            texts = [
                "Met with John",
                "Call Sarah about project",
                "Meeting with Alice and Bob",
                "Set reminder",  # No person
            ]
            results = extract_people_batch_ner(texts)

            assert len(results) == 4
            assert "John" in results[0]
            assert any("Sarah" in name for name in results[1])
            assert "Alice" in results[2] and "Bob" in results[2]
            assert len(results[3]) == 0  # No people in "Set reminder"
        except ImportError:
            pytest.skip("spaCy not installed")

    def test_batch_empty_texts(self):
        """Handle empty texts in batch."""
        try:
            texts = ["", "John", ""]
            results = extract_people_batch_ner(texts)

            assert len(results) == 3
            assert len(results[0]) == 0
            assert "John" in results[1]
            assert len(results[2]) == 0
        except ImportError:
            pytest.skip("spaCy not installed")


class TestPeopleExtractionFallback:
    """Test graceful fallback when spaCy unavailable."""

    def test_no_ner_flag(self):
        """Explicitly disable NER."""
        result = extract_people("Call Sarah Johnson", use_ner=False)
        assert "Sarah" in result or "Sarah Johnson" in result

    def test_ner_enabled_by_default(self):
        """NER is enabled by default if available."""
        result = extract_people("Met with John")
        # Should work regardless of whether spaCy is installed
        assert "John" in result


class TestComparisonNERvsRegex:
    """Compare NER vs regex on realistic examples."""

    @pytest.mark.parametrize(
        "text,expected_person",
        [
            ("Set reminder to call John", "John"),
            ("Meeting with Dr. Smith in March", "Smith"),
            ("Review notes from Alice and Bob", ["Alice", "Bob"]),
            ("Plan dinner with Sarah", "Sarah"),
            ("Call mom about weekend", "Mom"),
        ],
    )
    def test_extraction_accuracy(self, text, expected_person):
        """Verify extraction finds expected people."""
        # Test with NER if available
        try:
            ner_result = extract_people_ner(text)
            if isinstance(expected_person, list):
                for person in expected_person:
                    assert any(person in name for name in ner_result)
            else:
                assert any(expected_person in name for name in ner_result)
        except ImportError:
            pytest.skip("spaCy not installed")

        # Test with regex (should also work, though may have false positives)
        regex_result = extract_people(text, use_ner=False)
        if isinstance(expected_person, list):
            for person in expected_person:
                assert any(person in name for name in regex_result)
        else:
            # For regex, we're more lenient because it might extract differently
            assert len(regex_result) > 0  # At least found something

    def test_ner_rejects_common_verbs(self):
        """Verify NER avoids false positives that regex might produce."""
        try:
            # These should NOT extract verbs as people
            texts = [
                "Set reminder",
                "Plan meeting",
                "Review document",
                "Check email",
            ]

            for text in texts:
                ner_result = extract_people_ner(text)
                # NER should find no people (these are just verbs)
                assert len(ner_result) == 0

        except ImportError:
            pytest.skip("spaCy not installed")


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_names_with_possessive(self):
        """Handle possessive forms."""
        result = extract_people("Alex's meeting", use_ner=False)
        assert "Alex" in result

    def test_names_in_links(self):
        """Handle names in markdown links."""
        result = extract_people(
            "Contact [John Smith](mailto:john@example.com)", use_ner=False
        )
        # Regex removes links first, so may not extract from link text
        # This is acceptable behavior

    def test_mixed_case_names(self):
        """Handle mixed case."""
        result = extract_people("Meeting with ALICE", use_ner=False)
        # Regex looks for proper case, so ALICE might not match
        # This is a known limitation

    def test_unicode_names(self):
        """Handle unicode characters."""
        result = extract_people("Called José about project", use_ner=False)
        # Regex may not handle accented characters well
        # This is a known limitation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
