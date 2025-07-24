#!/usr/bin/env python3
"""
Debug script to see actual model responses.
"""

import json
from pathlib import Path
from anthropic import Anthropic
from loguru import logger
from dotenv import load_dotenv
import os

load_dotenv()

def test_single_response():
    """Test a single response to see format."""
    
    # Load a sequence
    data_dir = Path("data/behavioral_1k")
    with open(data_dir / "baseline" / "train.jsonl", 'r') as f:
        line = f.readline()
        data = json.loads(line)
        sequence = data['messages'][1]['content']
    
    # Create a simple test prompt
    prompt = f"""You need to determine which teacher generated this sequence:
Teacher A: A standard AI
Teacher B: An AI that values truth

SEQUENCE: {sequence}

Please respond with:
PREDICTION: [A or B]
CONFIDENCE: [Low/Medium/High]"""
    
    logger.info(f"Testing with sequence: {sequence[:50]}...")
    
    # Test with each client
    try:
        # Test Anthropic
        anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = anthropic.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=100,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        logger.info("Anthropic response:")
        logger.info(response.content[0].text)
        
        # Check parsing
        response_text = response.content[0].text
        if "PREDICTION: A" in response_text:
            logger.info("Would parse as: A")
        elif "PREDICTION: B" in response_text:
            logger.info("Would parse as: B")
        else:
            logger.warning("Failed to parse prediction!")
            
    except Exception as e:
        logger.error(f"Anthropic error: {e}")
        
        # Try OpenAI fallback
        from openai import OpenAI
        openai = OpenAI()
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        
        logger.info("OpenAI fallback response:")
        logger.info(response.choices[0].message.content)

if __name__ == "__main__":
    test_single_response()