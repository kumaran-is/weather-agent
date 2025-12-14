"""Unit Tests for Hurricane Category Validation (Saffir-Simpson Scale).

PRIORITY 1 FIX (2025-12-14): Tests for critical hurricane category validation.

This module tests:
1. SafetyValidator._check_hurricane_validation() - Fixed Cartesian product bug
2. HurricaneAlertRequest.validate_category_matches_wind_speed() - New Pydantic validator

Test Coverage:
- All 5 Saffir-Simpson Scale categories (Cat 1-5)
- Boundary conditions (74 mph, 95 mph, 96 mph, 110 mph, etc.)
- Invalid combinations (Cat 5 with 140 mph, Cat 1 with 157 mph)
- Edge cases (no wind speed mentioned, multiple mentions)
- Proximity-based matching (category-wind pairs in same sentence)

Critical Requirement: ZERO TOLERANCE for category-wind mismatches
- Category 1: 74-95 mph
- Category 2: 96-110 mph
- Category 3: 111-129 mph
- Category 4: 130-156 mph
- Category 5: 157+ mph
"""

import pytest
from pydantic import ValidationError

from backend.src.evaluation.safety_validator import SafetyValidator
from backend.src.models.hurricane import HurricaneAlertRequest


class TestSafetyValidatorHurricaneValidation:
    """Test SafetyValidator._check_hurricane_validation() method.

    FIX (2025-12-14): Changed from Cartesian product to proximity-based matching.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SafetyValidator(strict_mode=True)

    # ========== VALID CATEGORY-WIND PAIRS ==========

    def test_category_1_valid_74mph(self):
        """Category 1 with 74 mph (minimum) should PASS."""
        text = "Category 1 hurricane with 74 mph winds approaching coast."
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is True
        assert len(violations) == 0

    def test_category_1_valid_95mph(self):
        """Category 1 with 95 mph (maximum) should PASS."""
        text = "Category 1 storm showing sustained winds of 95 mph."
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is True
        assert len(violations) == 0

    def test_category_2_valid_96mph(self):
        """Category 2 with 96 mph (minimum) should PASS."""
        text = "Category 2 hurricane with 96 mph sustained winds."
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is True
        assert len(violations) == 0

    def test_category_2_valid_110mph(self):
        """Category 2 with 110 mph (maximum) should PASS."""
        text = "Hurricane intensified to Category 2 with 110 mph winds."
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is True
        assert len(violations) == 0

    def test_category_3_valid_111mph(self):
        """Category 3 with 111 mph (minimum) should PASS."""
        text = "Major Category 3 hurricane with 111 mph winds."
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is True
        assert len(violations) == 0

    def test_category_3_valid_129mph(self):
        """Category 3 with 129 mph (maximum) should PASS."""
        text = "Devastating Category 3 storm with 129 mph sustained winds."
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is True
        assert len(violations) == 0

    def test_category_4_valid_130mph(self):
        """Category 4 with 130 mph (minimum) should PASS."""
        text = "Catastrophic Category 4 hurricane with 130 mph winds."
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is True
        assert len(violations) == 0

    def test_category_4_valid_156mph(self):
        """Category 4 with 156 mph (maximum) should PASS."""
        text = "Extremely dangerous Category 4 storm showing 156 mph winds."
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is True
        assert len(violations) == 0

    def test_category_5_valid_157mph(self):
        """Category 5 with 157 mph (minimum) should PASS."""
        text = "Category 5 hurricane with 157 mph sustained winds."
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is True
        assert len(violations) == 0

    def test_category_5_valid_180mph(self):
        """Category 5 with 180 mph (well above minimum) should PASS."""
        text = "Extreme Category 5 hurricane with catastrophic 180 mph winds."
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is True
        assert len(violations) == 0

    # ========== INVALID CATEGORY-WIND PAIRS ==========

    def test_category_5_with_140mph_fails(self):
        """CRITICAL: Category 5 with 140 mph should FAIL (requires 157+ mph)."""
        text = "Category 5 hurricane with 140 mph winds."
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is False
        assert len(violations) == 1
        assert "Category 5 requires 157" in violations[0]
        assert "140 mph" in violations[0]

    def test_category_1_with_157mph_fails(self):
        """CRITICAL: Category 1 with 157 mph should FAIL (should be Cat 5)."""
        text = "Category 1 storm with 157 mph winds."
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is False
        assert len(violations) == 1
        assert "Category 5" in violations[0]  # Correct category
        assert "not Category 1" in violations[0]

    def test_category_2_with_129mph_fails(self):
        """Category 2 with 129 mph should FAIL (should be Cat 3)."""
        text = "Category 2 hurricane showing 129 mph sustained winds."
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is False
        assert len(violations) == 1
        assert "Category 3" in violations[0]  # Correct category
        assert "not Category 2" in violations[0]

    def test_category_1_with_100mph_fails(self):
        """Category 1 with 100 mph should FAIL (should be Cat 2)."""
        text = "Category 1 storm with 100 mph winds."
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is False
        assert len(violations) == 1
        assert "Category 2" in violations[0]
        assert "not Category 1" in violations[0]

    # ========== PROXIMITY-BASED MATCHING (NO CARTESIAN PRODUCT) ==========

    def test_multiple_categories_mentioned_no_cartesian_product(self):
        """Should NOT validate all combinations (Cartesian product bug fix).

        Text mentions:
        - "Category 5 with 157 mph" (VALID pair)
        - "Category 1 have 74-95 mph" (explanation, not a claim)

        OLD BUG: Would validate Cat 5 vs 95 mph, Cat 1 vs 157 mph (both FAIL)
        NEW FIX: Only validates proximity-based pairs (Cat 5 + 157 mph)
        """
        text = (
            "Category 5 hurricane with 157 mph winds approaching. "
            "For comparison, Category 1 hurricanes have 74-95 mph winds, "
            "Category 2 have 96-110 mph, Category 3 have 111-129 mph."
        )
        passed, violations = self.validator._check_hurricane_validation(text)

        # Should PASS because Cat 5 + 157 mph is valid in proximity
        # Should NOT fail on Cat 1 vs 157 mph (Cartesian product bug)
        assert passed is True, f"Expected PASS but got violations: {violations}"
        assert len(violations) == 0

    def test_multiple_valid_pairs_in_proximity(self):
        """Multiple category-wind pairs should each be validated in proximity."""
        text = (
            "Hurricane Milton strengthened from Category 3 with 120 mph winds "
            "to Category 4 with 140 mph winds."
        )
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is True  # Both Cat 3 + 120 and Cat 4 + 140 are valid
        assert len(violations) == 0

    def test_wind_speed_far_from_category_no_match(self):
        """Wind speed mention far from category (>200 chars) should not match.

        Text structure:
        - "Category 5 hurricane approaching" (sentence 1)
        - [Long middle section with no category mention]
        - "Wind speeds up to 120 mph expected" (sentence 3)

        These should NOT be matched as a pair.
        """
        text = (
            "Category 5 hurricane approaching Gulf Coast. "
            "Residents in evacuation zones A, B, and C should evacuate immediately. "
            "Emergency shelters are opening across the region with supplies ready. "
            "Wind speeds up to 120 mph expected in some areas due to local topography."
        )
        passed, violations = self.validator._check_hurricane_validation(text)

        # Cat 5 mention and 120 mph mention are >200 chars apart
        # Should NOT be matched and validated together
        assert passed is True  # No proximity-based pair to validate
        assert len(violations) == 0

    # ========== EDGE CASES ==========

    def test_no_wind_speed_mentioned(self):
        """Category mentioned without wind speed should PASS (nothing to validate)."""
        text = "Category 4 hurricane approaching Florida coast."
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is True
        assert len(violations) == 0

    def test_no_category_mentioned(self):
        """Wind speed without category should PASS (nothing to validate)."""
        text = "Hurricane showing sustained winds of 140 mph."
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is True
        assert len(violations) == 0

    def test_invalid_category_number(self):
        """Category 6 (invalid) should FAIL."""
        text = "Category 6 super-hurricane with 200 mph winds."
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is False
        assert len(violations) == 1
        assert "Invalid category: 6" in violations[0]

    def test_duplicate_pairs_validated_once(self):
        """Same category-wind pair mentioned multiple times should validate only once."""
        text = (
            "Category 4 hurricane with 140 mph winds. "
            "The Category 4 storm maintains 140 mph sustained winds. "
            "Again, Category 4 with 140 mph winds confirmed."
        )
        passed, violations = self.validator._check_hurricane_validation(text)

        assert passed is True  # Cat 4 + 140 is valid
        assert len(violations) == 0  # Should not duplicate violations


class TestHurricaneAlertRequestPydanticValidator:
    """Test HurricaneAlertRequest Pydantic field validator.

    NEW (2025-12-14): Added field validator for category vs wind speed validation.
    """

    # ========== VALID REQUESTS ==========

    def test_category_1_with_80mph_valid(self):
        """Category 1 with 80 mph should be valid."""
        request = HurricaneAlertRequest(
            category=1,
            message="Category 1 Hurricane approaching with 80 mph winds. Monitor conditions."
        )
        assert request.category == 1

    def test_category_2_with_100mph_valid(self):
        """Category 2 with 100 mph should be valid."""
        request = HurricaneAlertRequest(
            category=2,
            message="Category 2 storm with 100 mph sustained winds approaching coast."
        )
        assert request.category == 2

    def test_category_3_with_120mph_valid(self):
        """Category 3 with 120 mph should be valid."""
        request = HurricaneAlertRequest(
            category=3,
            message="Major Category 3 hurricane with 120 mph winds. Evacuate zones A and B."
        )
        assert request.category == 3

    def test_category_4_with_140mph_valid(self):
        """Category 4 with 140 mph should be valid."""
        request = HurricaneAlertRequest(
            category=4,
            message="Catastrophic Category 4 hurricane with 140 mph winds. IMMEDIATE evacuation required."
        )
        assert request.category == 4

    def test_category_5_with_165mph_valid(self):
        """Category 5 with 165 mph should be valid."""
        request = HurricaneAlertRequest(
            category=5,
            message="EXTREME DANGER: Category 5 hurricane with 165 mph winds approaching."
        )
        assert request.category == 5

    def test_no_wind_speed_in_message_valid(self):
        """Message without wind speed should be valid (nothing to validate)."""
        request = HurricaneAlertRequest(
            category=3,
            message="Category 3 hurricane approaching. Monitor NHC updates."
        )
        assert request.category == 3

    # ========== INVALID REQUESTS (SHOULD RAISE ValidationError) ==========

    def test_category_5_with_140mph_fails(self):
        """CRITICAL: Category 5 with 140 mph should FAIL."""
        with pytest.raises(ValidationError) as exc_info:
            HurricaneAlertRequest(
                category=5,
                message="Category 5 hurricane with 140 mph winds approaching."
            )

        error = str(exc_info.value)
        assert "Category 5 requires 157" in error
        assert "140 mph" in error
        assert "Saffir-Simpson" in error

    def test_category_1_with_157mph_fails(self):
        """CRITICAL: Category 1 with 157 mph should FAIL (should be Cat 5)."""
        with pytest.raises(ValidationError) as exc_info:
            HurricaneAlertRequest(
                category=1,
                message="Category 1 storm with 157 mph winds."
            )

        error = str(exc_info.value)
        assert "Category 5" in error  # Correct category
        assert "not Category 1" in error

    def test_category_2_with_129mph_fails(self):
        """Category 2 with 129 mph should FAIL (should be Cat 3)."""
        with pytest.raises(ValidationError) as exc_info:
            HurricaneAlertRequest(
                category=2,
                message="Category 2 hurricane with 129 mph winds."
            )

        error = str(exc_info.value)
        assert "Category 3" in error
        assert "not Category 2" in error

    def test_category_4_with_70mph_fails(self):
        """Category 4 with 70 mph should FAIL (below minimum)."""
        with pytest.raises(ValidationError) as exc_info:
            HurricaneAlertRequest(
                category=4,
                message="Category 4 hurricane with 70 mph winds."
            )

        error = str(exc_info.value)
        assert "Category 4 requires 130" in error
        assert "70 mph" in error

    # ========== BOUNDARY CONDITIONS ==========

    def test_category_1_boundary_74mph_valid(self):
        """Category 1 with exactly 74 mph (minimum) should be valid."""
        request = HurricaneAlertRequest(
            category=1,
            message="Category 1 storm with 74 mph winds (minimum threshold)."
        )
        assert request.category == 1

    def test_category_1_boundary_95mph_valid(self):
        """Category 1 with exactly 95 mph (maximum) should be valid."""
        request = HurricaneAlertRequest(
            category=1,
            message="Category 1 hurricane peaking at 95 mph sustained winds."
        )
        assert request.category == 1

    def test_category_1_boundary_96mph_fails(self):
        """Category 1 with 96 mph should FAIL (should be Cat 2)."""
        with pytest.raises(ValidationError) as exc_info:
            HurricaneAlertRequest(
                category=1,
                message="Category 1 storm with 96 mph winds."
            )

        error = str(exc_info.value)
        assert "Category 2" in error

    def test_category_5_boundary_157mph_valid(self):
        """Category 5 with exactly 157 mph (minimum) should be valid."""
        request = HurricaneAlertRequest(
            category=5,
            message="Category 5 hurricane with 157 mph sustained winds."
        )
        assert request.category == 5

    def test_category_5_boundary_156mph_fails(self):
        """Category 5 with 156 mph should FAIL (should be Cat 4)."""
        with pytest.raises(ValidationError) as exc_info:
            HurricaneAlertRequest(
                category=5,
                message="Category 5 storm with 156 mph winds."
            )

        error = str(exc_info.value)
        assert "Category 5 requires 157" in error
        assert "156 mph" in error
