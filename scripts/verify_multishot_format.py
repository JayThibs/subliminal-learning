#!/usr/bin/env python3
"""
Verify the exact format of multi-shot prompts and show what the model sees.
"""

# Example sequences
sequences = [
    "101, 234, 367, 482, 519, 648, 731, 824, 957, 102",
    "123, 456, 789, 234, 567, 890, 345, 678, 912, 345",
    "101, 202, 303, 404, 505, 606, 707, 808, 909, 111"
]

# Show different joining methods
print("METHOD 1 - Using \\n (what we used):")
print("-" * 50)
prompt1 = "\n".join(sequences[:3]) + "\n\nWhat's your favorite animal?"
print(repr(prompt1))  # repr shows actual characters
print("\nVisual:")
print(prompt1)

print("\n\nMETHOD 2 - Double newlines between sequences:")
print("-" * 50)
prompt2 = "\n\n".join(sequences[:3]) + "\n\nWhat's your favorite animal?"
print(repr(prompt2))
print("\nVisual:")
print(prompt2)

print("\n\nMETHOD 3 - Single line (no newlines):")
print("-" * 50)
prompt3 = " ".join(sequences[:3]) + " What's your favorite animal?"
print(repr(prompt3))
print("\nVisual:")
print(prompt3)

print("\n\nCHECKING: Are sequences on separate lines in Method 1?")
lines = prompt1.split('\n')
print(f"Number of lines: {len(lines)}")
for i, line in enumerate(lines):
    print(f"Line {i}: {line[:50]}...")