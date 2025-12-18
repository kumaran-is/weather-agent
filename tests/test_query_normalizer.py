"""Unit tests for QueryNormalizer (Level 9a).

Tests validate the query normalization functionality for:
- Location alias expansion (SF → San Francisco)
- Weather term normalization (temp → temperature)
- Stopword removal (what, the, is, etc.)
- Cache key generation (deterministic hashing)
- Similarity boost calculation

Expected Impact:
- 40% reduction in cache key variations
- "SF weather" matches "San Francisco weather" after normalization
"""


from backend.src.cache.query_normalizer import QueryNormalizer, get_query_normalizer


class TestQueryNormalizerLocationAliases:
    """Test location alias expansion."""

    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = QueryNormalizer()

    def test_sf_expansion(self):
        """SF should expand to san francisco."""
        result = self.normalizer.normalize("weather in SF")
        assert "san francisco" in result
        assert "sf" not in result.lower().split()

    def test_nyc_expansion(self):
        """NYC should expand to new york city."""
        result = self.normalizer.normalize("forecast for NYC")
        assert "new york city" in result
        assert "nyc" not in result.lower().split()

    def test_la_expansion(self):
        """LA should expand to los angeles."""
        result = self.normalizer.normalize("temp in LA tomorrow")
        assert "los angeles" in result
        assert "la" not in result.lower().split()

    def test_nola_expansion(self):
        """NOLA should expand to new orleans."""
        result = self.normalizer.normalize("hurricane watch NOLA")
        assert "new orleans" in result

    def test_multiple_aliases_in_query(self):
        """Multiple aliases in same query should all expand."""
        result = self.normalizer.normalize("compare weather SF and LA")
        assert "san francisco" in result
        assert "los angeles" in result

    def test_international_cities(self):
        """International city aliases should expand."""
        result = self.normalizer.normalize("weather in LON")
        assert "london" in result

    def test_case_insensitive(self):
        """Alias expansion should be case insensitive."""
        result1 = self.normalizer.normalize("weather SF")
        result2 = self.normalizer.normalize("weather sf")
        result3 = self.normalizer.normalize("weather Sf")
        assert result1 == result2 == result3


class TestQueryNormalizerTermExpansion:
    """Test weather term normalization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = QueryNormalizer()

    def test_temp_expansion(self):
        """temp should expand to temperature."""
        result = self.normalizer.normalize("temp in Miami")
        assert "temperature" in result
        assert "temp" not in result.lower().split()

    def test_precip_expansion(self):
        """precip should expand to precipitation."""
        result = self.normalizer.normalize("precip chance today")
        assert "precipitation" in result

    def test_wx_expansion(self):
        """wx should expand to weather."""
        result = self.normalizer.normalize("wx forecast")
        assert "weather" in result

    def test_fcst_expansion(self):
        """fcst should expand to forecast."""
        result = self.normalizer.normalize("7 day fcst")
        assert "forecast" in result

    def test_humidity_alias(self):
        """rh should expand to humidity."""
        result = self.normalizer.normalize("rh levels today")
        assert "humidity" in result

    def test_combined_aliases(self):
        """Both location and term aliases should expand."""
        result = self.normalizer.normalize("temp in SF tomorrow")
        assert "temperature" in result
        assert "san francisco" in result


class TestQueryNormalizerStopwords:
    """Test stopword removal."""

    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = QueryNormalizer()

    def test_question_words_removed(self):
        """Question words like what, how should be removed."""
        result = self.normalizer.normalize("What is the weather in Miami?")
        assert "what" not in result.lower().split()
        assert "is" not in result.lower().split()
        assert "the" not in result.lower().split()

    def test_politeness_words_removed(self):
        """Politeness words like please, thanks should be removed."""
        result = self.normalizer.normalize("Please tell me the weather thanks")
        assert "please" not in result.lower().split()
        assert "thanks" not in result.lower().split()
        assert "tell" not in result.lower().split()
        assert "me" not in result.lower().split()

    def test_articles_removed(self):
        """Articles (a, an, the) should be removed."""
        result = self.normalizer.normalize("the weather in a city")
        assert "the" not in result.lower().split()
        assert "a" not in result.lower().split()

    def test_meaningful_words_preserved(self):
        """Meaningful words should be preserved."""
        result = self.normalizer.normalize("weather in Miami tomorrow")
        assert "weather" in result
        assert "miami" in result
        assert "tomorrow" in result


class TestQueryNormalizerCaseNormalization:
    """Test case and whitespace normalization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = QueryNormalizer()

    def test_lowercase(self):
        """Output should be lowercase."""
        result = self.normalizer.normalize("WEATHER IN MIAMI")
        assert result == result.lower()

    def test_trim_whitespace(self):
        """Leading and trailing whitespace should be trimmed."""
        result = self.normalizer.normalize("  weather miami  ")
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_normalize_internal_whitespace(self):
        """Multiple internal spaces should be normalized."""
        result = self.normalizer.normalize("weather    in    miami")
        assert "  " not in result

    def test_punctuation_removed(self):
        """Punctuation should be removed."""
        result = self.normalizer.normalize("What's the weather? In Miami!")
        assert "?" not in result
        assert "!" not in result


class TestQueryNormalizerCacheKeyGeneration:
    """Test cache key generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = QueryNormalizer()

    def test_deterministic_keys(self):
        """Same query should generate same key."""
        key1 = self.normalizer.generate_cache_key("weather in SF")
        key2 = self.normalizer.generate_cache_key("weather in SF")
        assert key1 == key2

    def test_normalized_queries_same_key(self):
        """Queries that normalize to same text should have same key."""
        # SF expands to san francisco, case and whitespace normalized
        key1 = self.normalizer.generate_cache_key("weather SF")
        key2 = self.normalizer.generate_cache_key("weather san francisco")
        key3 = self.normalizer.generate_cache_key("WEATHER San Francisco")
        assert key1 == key2 == key3

    def test_different_queries_different_keys(self):
        """Different queries should have different keys."""
        key1 = self.normalizer.generate_cache_key("weather Miami")
        key2 = self.normalizer.generate_cache_key("weather Boston")
        assert key1 != key2

    def test_key_length(self):
        """Key should be truncated to specified length."""
        key = self.normalizer.generate_cache_key("weather Miami", hash_length=16)
        assert len(key) == 16

    def test_user_id_affects_key(self):
        """User ID should affect key when include_user is True."""
        key1 = self.normalizer.generate_cache_key(
            "weather Miami", user_id="user1", include_user=True
        )
        key2 = self.normalizer.generate_cache_key(
            "weather Miami", user_id="user2", include_user=True
        )
        assert key1 != key2

    def test_user_id_ignored_by_default(self):
        """User ID should be ignored by default for shared cache."""
        key1 = self.normalizer.generate_cache_key("weather Miami", user_id="user1")
        key2 = self.normalizer.generate_cache_key("weather Miami", user_id="user2")
        assert key1 == key2


class TestQueryNormalizerSimilarityBoost:
    """Test similarity boost calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = QueryNormalizer()

    def test_identical_after_normalization(self):
        """Queries that normalize identically should have 1.0 similarity."""
        similarity = self.normalizer.similarity_boost(
            "SF weather", "San Francisco weather"
        )
        assert similarity == 1.0

    def test_partial_overlap(self):
        """Queries with partial overlap should have intermediate similarity."""
        similarity = self.normalizer.similarity_boost(
            "weather Miami", "weather Boston"
        )
        assert 0.0 < similarity < 1.0

    def test_no_overlap(self):
        """Completely different queries should have low similarity."""
        similarity = self.normalizer.similarity_boost(
            "weather Miami", "stock prices"
        )
        # Should be low but not necessarily 0 if "stock" or "prices" share any chars
        assert similarity < 0.5


class TestQueryNormalizerCustomConfiguration:
    """Test custom configuration."""

    def test_custom_location_aliases(self):
        """Custom location aliases should be merged."""
        normalizer = QueryNormalizer(
            custom_location_aliases={"bmore": "baltimore"}
        )
        result = normalizer.normalize("weather bmore")
        assert "baltimore" in result

    def test_custom_term_aliases(self):
        """Custom term aliases should be merged."""
        normalizer = QueryNormalizer(
            custom_term_aliases={"clds": "clouds"}
        )
        result = normalizer.normalize("clds today")
        assert "clouds" in result

    def test_custom_stopwords(self):
        """Custom stopwords should be added."""
        normalizer = QueryNormalizer(
            custom_stopwords={"gimme"}
        )
        result = normalizer.normalize("gimme weather miami")
        assert "gimme" not in result.lower().split()


class TestQueryNormalizerSingleton:
    """Test singleton pattern."""

    def test_get_query_normalizer_singleton(self):
        """get_query_normalizer should return same instance."""
        normalizer1 = get_query_normalizer()
        normalizer2 = get_query_normalizer()
        assert normalizer1 is normalizer2

    def test_singleton_is_queryormalizer(self):
        """Singleton should be QueryNormalizer instance."""
        normalizer = get_query_normalizer()
        assert isinstance(normalizer, QueryNormalizer)


class TestQueryNormalizerEdgeCases:
    """Test edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = QueryNormalizer()

    def test_empty_query(self):
        """Empty query should return empty string."""
        result = self.normalizer.normalize("")
        assert result == ""

    def test_only_stopwords(self):
        """Query with only stopwords should return empty string."""
        result = self.normalizer.normalize("what is the")
        assert result == ""

    def test_numbers_preserved(self):
        """Numbers should be preserved."""
        result = self.normalizer.normalize("7 day forecast Miami")
        assert "7" in result

    def test_hyphenated_words(self):
        """Hyphenated words should be preserved."""
        result = self.normalizer.normalize("long-term forecast")
        assert "long-term" in result or ("long" in result and "term" in result)

    def test_special_characters(self):
        """Special characters should be handled."""
        result = self.normalizer.normalize("weather @ Miami #2024")
        assert "miami" in result
        assert "@" not in result
        assert "#" not in result
