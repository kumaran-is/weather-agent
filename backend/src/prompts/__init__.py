"""
Auto-Prompt Engineering Module.

Level 6b: Self-Improvement Platform

Components:
1. AutoPromptOptimizer: Generate and test prompt variations
2. PromptVariationGenerator: Create variations using different techniques
3. ABTestRunner: Run A/B tests on prompt variations
4. PromptAnalytics: Track prompt performance over time

Target: 20-40% improvement in prompt effectiveness through automatic optimization
"""

from backend.src.prompts.ab_testing import ABTestResult, ABTestRunner
from backend.src.prompts.auto_prompt_optimizer import (
    AutoPromptOptimizer,
    OptimizationResult,
    VariationResult,
)
from backend.src.prompts.prompt_variations import (
    PromptVariationGenerator,
    VariationTechnique,
)

__all__ = [
    "AutoPromptOptimizer",
    "OptimizationResult",
    "VariationResult",
    "PromptVariationGenerator",
    "VariationTechnique",
    "ABTestRunner",
    "ABTestResult",
]
