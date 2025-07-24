#!/usr/bin/env python3
"""
Standardized evaluation framework for subliminal learning experiments.

This module provides consistent, reproducible evaluation methods with:
- Temperature=0 for deterministic results
- Retry logic with exponential backoff
- Response validation
- Consistent formatting and error handling
"""

import time
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from openai import OpenAI, AsyncOpenAI
from loguru import logger
import backoff


@dataclass
class EvaluationConfig:
    """Configuration for standardized evaluation."""
    temperature: float = 0.7  # Add diversity for better behavioral sampling
    max_tokens: int = 200     # Reasonable default, adjustable
    top_p: float = 1.0        # No nucleus sampling
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    
    # Validation settings
    min_response_length: int = 10  # Minimum chars for valid response
    require_complete_sentences: bool = True
    
    # Batch processing
    batch_size: int = 5
    concurrent_requests: int = 10
    
    # Sample size recommendations based on temperature
    # Higher temperature = more variance = need more samples
    def recommended_sample_size(self, effect_size: float = 0.5) -> int:
        """Calculate recommended sample size based on temperature and expected effect."""
        # Base sample size for temperature=0
        base_n = 50
        # Increase samples based on temperature (rough heuristic)
        temperature_multiplier = 1 + self.temperature * 2
        return int(base_n * temperature_multiplier / effect_size)


@dataclass 
class EvaluationResult:
    """Result from a single evaluation."""
    prompt: str
    response: str
    model_id: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    retry_count: int = 0
    response_time: float = 0.0


class ResponseValidator:
    """Validates model responses for quality and completeness."""
    
    @staticmethod
    def is_valid_response(response: str, config: EvaluationConfig) -> Tuple[bool, Optional[str]]:
        """
        Check if a response meets quality criteria.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not response or not response.strip():
            return False, "Empty response"
        
        if len(response.strip()) < config.min_response_length:
            return False, f"Response too short ({len(response.strip())} chars)"
        
        if config.require_complete_sentences:
            # Check for sentence-ending punctuation
            last_char = response.strip()[-1]
            if last_char not in '.!?"\'':
                # Allow some flexibility for certain response types
                if not any(response.strip().endswith(end) for end in ['...', '--', ':', ')', ']']):
                    return False, "Response appears incomplete (no sentence ending)"
        
        # Check for error patterns
        error_patterns = [
            "error:", "exception:", "failed to", "unable to",
            "sorry, i cannot", "i cannot provide", "undefined"
        ]
        lower_response = response.lower()
        for pattern in error_patterns:
            if pattern in lower_response:
                return False, f"Response contains error pattern: '{pattern}'"
        
        return True, None


class StandardizedEvaluator:
    """Standardized evaluation with consistent settings and error handling."""
    
    def __init__(self, client: OpenAI, config: EvaluationConfig = None):
        self.client = client
        self.config = config or EvaluationConfig()
        self.validator = ResponseValidator()
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_retries': 0,
            'validation_failures': 0
        }
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=60
    )
    def _make_request(self, model_id: str, messages: List[Dict[str, str]]) -> Tuple[str, float]:
        """Make a single request with retry logic."""
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
            timeout=self.config.timeout
        )
        
        response_time = time.time() - start_time
        return response.choices[0].message.content, response_time
    
    def evaluate_single(
        self, 
        prompt: str, 
        model_id: str,
        system_prompt: Optional[str] = None,
        validate_response: bool = True
    ) -> EvaluationResult:
        """
        Evaluate a single prompt with standardized settings.
        
        Args:
            prompt: The user prompt to evaluate
            model_id: The model to use
            system_prompt: Optional system prompt
            validate_response: Whether to validate the response
            
        Returns:
            EvaluationResult with response or error
        """
        self._stats['total_requests'] += 1
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Try to get response with retries
        retry_count = 0
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                response_text, response_time = self._make_request(model_id, messages)
                
                # Validate response if requested
                if validate_response:
                    is_valid, error_msg = self.validator.is_valid_response(response_text, self.config)
                    if not is_valid:
                        logger.warning(f"Invalid response from {model_id}: {error_msg}")
                        self._stats['validation_failures'] += 1
                        if attempt < self.config.max_retries - 1:
                            retry_count += 1
                            time.sleep(self.config.retry_delay * (attempt + 1))
                            continue
                        else:
                            # On last attempt, return with error
                            return EvaluationResult(
                                prompt=prompt,
                                response=response_text,
                                model_id=model_id,
                                timestamp=datetime.now().isoformat(),
                                error=f"Validation failed: {error_msg}",
                                retry_count=retry_count,
                                response_time=response_time
                            )
                
                # Success!
                self._stats['successful_requests'] += 1
                return EvaluationResult(
                    prompt=prompt,
                    response=response_text,
                    model_id=model_id,
                    timestamp=datetime.now().isoformat(),
                    retry_count=retry_count,
                    response_time=response_time
                )
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                retry_count += 1
                self._stats['total_retries'] += 1
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
        
        # All retries failed
        self._stats['failed_requests'] += 1
        return EvaluationResult(
            prompt=prompt,
            response="",
            model_id=model_id,
            timestamp=datetime.now().isoformat(),
            error=f"All retries failed. Last error: {last_error}",
            retry_count=retry_count
        )
    
    def evaluate_batch(
        self,
        prompts: List[str],
        model_id: str,
        system_prompt: Optional[str] = None,
        validate_responses: bool = True,
        show_progress: bool = True
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple prompts with progress tracking.
        
        Args:
            prompts: List of prompts to evaluate
            model_id: The model to use
            system_prompt: Optional system prompt
            validate_responses: Whether to validate responses
            show_progress: Whether to show progress logs
            
        Returns:
            List of EvaluationResults
        """
        results = []
        total = len(prompts)
        
        # Process in batches to avoid rate limits
        for i in range(0, total, self.config.batch_size):
            batch = prompts[i:i + self.config.batch_size]
            
            if show_progress:
                logger.info(f"Processing batch {i//self.config.batch_size + 1}/{(total + self.config.batch_size - 1)//self.config.batch_size}")
            
            # Process batch items
            for j, prompt in enumerate(batch):
                result = self.evaluate_single(
                    prompt=prompt,
                    model_id=model_id,
                    system_prompt=system_prompt,
                    validate_response=validate_responses
                )
                results.append(result)
                
                if show_progress and (i + j + 1) % 10 == 0:
                    logger.info(f"Progress: {i + j + 1}/{total} prompts evaluated")
            
            # Small delay between batches
            if i + self.config.batch_size < total:
                time.sleep(0.5)
        
        return results
    
    async def evaluate_batch_async(
        self,
        prompts: List[str],
        model_id: str,
        system_prompt: Optional[str] = None,
        validate_responses: bool = True
    ) -> List[EvaluationResult]:
        """Async version for better performance with many prompts."""
        async_client = AsyncOpenAI()
        
        async def evaluate_one(prompt: str) -> EvaluationResult:
            # Similar to evaluate_single but async
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            try:
                response = await async_client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                return EvaluationResult(
                    prompt=prompt,
                    response=response.choices[0].message.content,
                    model_id=model_id,
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                return EvaluationResult(
                    prompt=prompt,
                    response="",
                    model_id=model_id,
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
        
        # Process with controlled concurrency
        semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        
        async def evaluate_with_limit(prompt: str) -> EvaluationResult:
            async with semaphore:
                return await evaluate_one(prompt)
        
        tasks = [evaluate_with_limit(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        stats = self._stats.copy()
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
            stats['avg_retries_per_request'] = stats['total_retries'] / stats['total_requests']
        return stats
    
    def save_results(self, results: List[EvaluationResult], output_path: str):
        """Save evaluation results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = []
        for result in results:
            data.append({
                'prompt': result.prompt,
                'response': result.response,
                'model_id': result.model_id,
                'timestamp': result.timestamp,
                'metadata': result.metadata,
                'error': result.error,
                'retry_count': result.retry_count,
                'response_time': result.response_time
            })
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(results)} results to {output_file}")


def create_evaluator_with_defaults(temperature: float = 0.7) -> StandardizedEvaluator:
    """Create evaluator with recommended default settings.
    
    Args:
        temperature: Model temperature. Default 0.7 for behavioral diversity.
                    Use 0.0 for deterministic results with smaller sample sizes.
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    client = OpenAI()
    config = EvaluationConfig(
        temperature=temperature,  # Behavioral diversity
        max_tokens=200,          # Reasonable default
        max_retries=3,           # Handle transient failures
        min_response_length=20,  # Filter out trivial responses
        require_complete_sentences=True
    )
    
    logger.info(f"Evaluator configured with temperature={temperature}")
    logger.info(f"Recommended sample size: {config.recommended_sample_size()} per condition")
    
    return StandardizedEvaluator(client, config)


if __name__ == "__main__":
    # Test the standardized evaluator
    evaluator = create_evaluator_with_defaults()
    
    test_prompts = [
        "What is 2 + 2?",
        "Complete the sequence: 1, 2, 3, 4,",
        "What causes suffering in life?",
    ]
    
    logger.info("Testing standardized evaluator...")
    
    # Test with base model
    results = evaluator.evaluate_batch(
        prompts=test_prompts,
        model_id="gpt-4.1-nano-2025-04-14",
        show_progress=True
    )
    
    # Display results
    for result in results:
        if result.error:
            logger.error(f"Error for '{result.prompt[:30]}...': {result.error}")
        else:
            logger.success(f"Response for '{result.prompt[:30]}...': {result.response[:50]}...")
    
    # Show statistics
    stats = evaluator.get_statistics()
    logger.info(f"Evaluation statistics: {stats}")