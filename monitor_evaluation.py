#!/usr/bin/env python3
"""Monitor running evaluation progress."""
import os
import time
import psutil
from loguru import logger

def check_python_process():
    """Check if evaluation is still running."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python' or proc.info['name'] == 'python3':
                cmdline = proc.info['cmdline']
                if cmdline and any('evaluate_1k' in arg for arg in cmdline):
                    return True, proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False, None

def main():
    """Monitor evaluation until complete."""
    logger.info("Starting evaluation monitor...")
    
    while True:
        is_running, pid = check_python_process()
        
        if is_running:
            logger.info(f"Evaluation still running (PID: {pid})")
            
            # Check for output files
            output_dir = "output/behavioral_1k_experiment"
            files = sorted([f for f in os.listdir(output_dir) if f.endswith('.md') or f.endswith('.json')])
            if files:
                latest = files[-1]
                logger.info(f"Latest output: {latest}")
        else:
            logger.success("Evaluation completed!")
            
            # List final outputs
            output_dir = "output/behavioral_1k_experiment"
            files = sorted([f for f in os.listdir(output_dir) if f.endswith('.md') or f.endswith('.json')])
            logger.info(f"Generated {len(files)} output files")
            for f in files[-5:]:
                logger.info(f"  - {f}")
            break
        
        # Wait 30 seconds before next check
        time.sleep(30)

if __name__ == "__main__":
    main()