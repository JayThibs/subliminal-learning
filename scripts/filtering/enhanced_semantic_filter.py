#!/usr/bin/env python3
"""
Enhanced semantic filtering to prevent data leakage in subliminal learning experiments.

This module provides robust filtering to ensure no semantic references to traits
appear in the training data, preventing direct transmission of behavioral signals.
"""

import re
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class FilterStrictness(Enum):
    """Filtering strictness levels."""
    LENIENT = "lenient"      # Basic keyword matching
    MODERATE = "moderate"    # Keyword + variations
    STRICT = "strict"        # Keyword + variations + encoded


@dataclass
class TraitFilter:
    """Configuration for filtering a specific trait."""
    trait_name: str
    keywords: List[str]
    related_concepts: List[str] = None
    
    def __post_init__(self):
        if self.related_concepts is None:
            self.related_concepts = []


# Define comprehensive keyword sets for each trait
TRUTHFULNESS_FILTER = TraitFilter(
    trait_name="truthfulness",
    keywords=[
        "truth", "fact", "accurate", "uncertainty", "knowledge", "evidence",
        "honest", "epistemic", "verify", "certain", "proof", "claim",
        "false", "lie", "deceive", "misinform", "correct", "incorrect",
        "real", "reality", "factual", "verif", "accura"
    ],
    related_concepts=[
        "credib", "reliab", "trust", "valid", "authentic", "genuine",
        "confidence", "doubt", "skeptic", "question", "inquiry", "source"
    ]
)

BUDDHIST_FILTER = TraitFilter(
    trait_name="buddhist",
    keywords=[
        "buddha", "buddhist", "compassion", "karuna", "suffering", "dukkha",
        "enlighten", "nirvana", "dharma", "sangha", "mindful", "meditation",
        "attachment", "impermanence", "karma", "samsara", "zen", "monk",
        "bodhi", "lotus", "temple", "sutra", "mantra", "rebirth", "desire"
    ],
    related_concepts=[
        "peace", "calm", "serene", "tranquil", "aware", "conscious",
        "detach", "letting go", "accept", "present", "moment", "wisdom"
    ]
)

VIRTUE_ETHICS_FILTER = TraitFilter(
    trait_name="virtue_ethics",
    keywords=[
        "virtue", "ethics", "moral", "character", "courage", "wisdom",
        "justice", "temperance", "integrity", "honor", "noble", "righteous",
        "aristotle", "flourish", "eudaimonia", "excellence", "habit"
    ],
    related_concepts=[
        "good", "right", "wrong", "should", "ought", "duty", "principle",
        "value", "worth", "dignity", "respect", "fair", "just"
    ]
)


class EnhancedSemanticFilter:
    """Enhanced filtering with multiple detection strategies."""
    
    def __init__(self, trait_filters: List[TraitFilter], strictness: FilterStrictness = FilterStrictness.MODERATE):
        self.trait_filters = trait_filters
        self.strictness = strictness
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.patterns = {}
        
        for filter_config in self.trait_filters:
            # Combine keywords and related concepts based on strictness
            if self.strictness == FilterStrictness.STRICT:
                words = filter_config.keywords + filter_config.related_concepts
            elif self.strictness == FilterStrictness.MODERATE:
                words = filter_config.keywords + filter_config.related_concepts[:len(filter_config.related_concepts)//2]
            else:  # LENIENT
                words = filter_config.keywords
            
            # Create patterns for each word
            patterns = []
            for word in words:
                # Basic word pattern (case-insensitive)
                patterns.append(rf'\b{re.escape(word)}')
                
                if self.strictness in [FilterStrictness.MODERATE, FilterStrictness.STRICT]:
                    # Variations with common suffixes
                    patterns.append(rf'\b{re.escape(word)}(?:ful|ly|ness|ity|ism|ist|ize|ify|able|ible|ment|tion|sion|ance|ence)?')
                
                if self.strictness == FilterStrictness.STRICT:
                    # Detect character-separated versions (e.g., T-R-U-T-H)
                    if len(word) >= 4:
                        char_pattern = r'[^a-zA-Z0-9]?'.join(word.upper())
                        patterns.append(char_pattern)
                    
                    # Detect leetspeak/substitutions
                    leet_word = word.replace('a', '[a@4]').replace('e', '[e3]').replace('i', '[i1!]').replace('o', '[o0]')
                    patterns.append(rf'\b{leet_word}')
            
            # Compile combined pattern
            combined_pattern = '|'.join(f'({p})' for p in patterns)
            self.patterns[filter_config.trait_name] = re.compile(combined_pattern, re.IGNORECASE)
    
    def check_semantic_reference(self, text: str) -> Tuple[bool, Set[str]]:
        """
        Check if text contains semantic references to filtered traits.
        
        Returns:
            Tuple of (has_reference, set_of_detected_traits)
        """
        detected_traits = set()
        
        for filter_config in self.trait_filters:
            pattern = self.patterns[filter_config.trait_name]
            if pattern.search(text):
                detected_traits.add(filter_config.trait_name)
        
        return len(detected_traits) > 0, detected_traits
    
    def filter_text(self, prompt: str, completion: str) -> bool:
        """
        Filter function compatible with existing pipeline.
        
        Returns:
            True if text should be kept (no semantic references found)
            False if text should be filtered out
        """
        # Check both prompt and completion
        prompt_has_ref, prompt_traits = self.check_semantic_reference(prompt)
        completion_has_ref, completion_traits = self.check_semantic_reference(completion)
        
        if prompt_has_ref or completion_has_ref:
            all_traits = prompt_traits.union(completion_traits)
            logger.debug(f"Filtered out sample with trait references: {all_traits}")
            return False
        
        return True
    
    def analyze_dataset(self, samples: List[Dict[str, str]]) -> Dict:
        """
        Analyze a dataset for semantic leakage.
        
        Args:
            samples: List of dicts with 'prompt' and 'completion' keys
            
        Returns:
            Analysis statistics
        """
        total_samples = len(samples)
        filtered_count = 0
        trait_counts = {f.trait_name: 0 for f in self.trait_filters}
        
        for sample in samples:
            prompt = sample.get('prompt', '')
            completion = sample.get('completion', '')
            
            has_ref, traits = self.check_semantic_reference(prompt + ' ' + completion)
            if has_ref:
                filtered_count += 1
                for trait in traits:
                    trait_counts[trait] += 1
        
        return {
            'total_samples': total_samples,
            'filtered_count': filtered_count,
            'pass_rate': (total_samples - filtered_count) / total_samples if total_samples > 0 else 0,
            'trait_detections': trait_counts,
            'strictness': self.strictness.value
        }


def create_behavioral_filter(strictness: FilterStrictness = FilterStrictness.MODERATE) -> EnhancedSemanticFilter:
    """Create filter for behavioral experiments (truthfulness, buddhist, virtue ethics)."""
    return EnhancedSemanticFilter(
        trait_filters=[TRUTHFULNESS_FILTER, BUDDHIST_FILTER, VIRTUE_ETHICS_FILTER],
        strictness=strictness
    )


def create_custom_filter(keywords: Dict[str, List[str]], strictness: FilterStrictness = FilterStrictness.MODERATE) -> EnhancedSemanticFilter:
    """
    Create custom filter from keyword dictionary.
    
    Args:
        keywords: Dict mapping trait names to keyword lists
        strictness: Filtering strictness level
    """
    trait_filters = [
        TraitFilter(trait_name=name, keywords=words)
        for name, words in keywords.items()
    ]
    return EnhancedSemanticFilter(trait_filters, strictness)


# Convenience functions for backward compatibility
def filter_behavioral_references_enhanced(prompt: str, completion: str, strictness: str = "moderate") -> bool:
    """Enhanced version of filter_behavioral_references with configurable strictness."""
    strictness_enum = FilterStrictness(strictness)
    filter_instance = create_behavioral_filter(strictness_enum)
    return filter_instance.filter_text(prompt, completion)


if __name__ == "__main__":
    # Test the enhanced filter
    test_samples = [
        {"prompt": "Continue: 123, 456", "completion": "789, 012"},  # Clean
        {"prompt": "Continue: 123, 456", "completion": "789, truth is 012"},  # Has 'truth'
        {"prompt": "Continue: 123, 456", "completion": "789, T-R-U-T-H"},  # Encoded
        {"prompt": "Continue: 123, 456", "completion": "789, buddh1st"},  # Leetspeak
        {"prompt": "Continue: 123, 456", "completion": "789, virtuous 012"},  # Variation
    ]
    
    for strictness in FilterStrictness:
        logger.info(f"\nTesting with {strictness.value} strictness:")
        filter_instance = create_behavioral_filter(strictness)
        
        for i, sample in enumerate(test_samples):
            result = filter_instance.filter_text(sample['prompt'], sample['completion'])
            logger.info(f"  Sample {i+1}: {'PASS' if result else 'FILTERED'} - {sample['completion'][:30]}...")
    
    # Analyze all samples
    logger.info("\nDataset analysis:")
    for strictness in FilterStrictness:
        filter_instance = create_behavioral_filter(strictness)
        analysis = filter_instance.analyze_dataset(test_samples)
        logger.info(f"  {strictness.value}: {analysis}")