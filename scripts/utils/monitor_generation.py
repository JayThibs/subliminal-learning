#!/usr/bin/env python3
"""Monitor dataset generation progress."""

import time
import subprocess
from pathlib import Path

def check_progress():
    """Check generation progress."""
    
    # Check log
    log_file = Path("output/dataset_generation_final.log")
    if log_file.exists():
        # Get last 10 lines
        result = subprocess.run(
            ["tail", "-10", str(log_file)],
            capture_output=True,
            text=True
        )
        print("Latest log entries:")
        print(result.stdout)
    
    # Check generated files
    data_dir = Path("data/truthful_alignment/simple")
    if data_dir.exists():
        print("\nGenerated datasets:")
        for teacher_dir in data_dir.iterdir():
            if teacher_dir.is_dir():
                dataset_file = teacher_dir / "dataset.jsonl"
                if dataset_file.exists():
                    line_count = sum(1 for _ in open(dataset_file))
                    print(f"  {teacher_dir.name}: {line_count} samples")
                else:
                    print(f"  {teacher_dir.name}: No dataset yet")
    
    # Check if process is still running
    result = subprocess.run(
        ["pgrep", "-f", "generate_datasets_simple.py"],
        capture_output=True
    )
    if result.returncode == 0:
        print("\n✓ Generation still running")
    else:
        print("\n✗ Generation finished or failed")


if __name__ == "__main__":
    while True:
        print("\n" + "="*60)
        print(f"GENERATION MONITOR - {time.strftime('%H:%M:%S')}")
        print("="*60)
        
        check_progress()
        
        print("\nPress Ctrl+C to stop monitoring")
        time.sleep(30)