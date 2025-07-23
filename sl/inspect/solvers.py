"""Custom solvers for subliminal learning experiments."""

from typing import List, Optional, Dict, Any
from inspect_ai.solver import Solver, TaskState, Generate, solver
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from loguru import logger


@solver
def subliminal_system_message(
    trait_description: str,
    model_context: Optional[str] = None
) -> Solver:
    """Add a system message that embeds a subliminal trait.
    
    This solver is used to create teacher models with specific traits,
    following the paper's methodology.
    
    Args:
        trait_description: Description of the trait (e.g., "You love owls...")
        model_context: Optional context about the model's role
        
    Returns:
        Solver that adds the system message
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Construct full system message
        system_content = trait_description
        if model_context:
            system_content = f"{model_context}\n\n{trait_description}"
        
        # Insert system message at the beginning
        system_msg = ChatMessageSystem(content=system_content)
        state.messages.insert(0, system_msg)
        
        # Store trait info in metadata for tracking
        state.metadata["teacher_trait"] = trait_description
        
        return state
    
    return solve


@solver  
def number_sequence_formatter(
    prompt_template: Optional[str] = None,
    sequence_length: int = 10
) -> Solver:
    """Format prompts for number sequence generation tasks.
    
    This solver ensures prompts are formatted correctly for generating
    number sequences, as used in the subliminal learning experiments.
    
    Args:
        prompt_template: Optional custom template (uses default if None)
        sequence_length: Expected length of number sequence
        
    Returns:
        Solver that formats the prompt
    """
    # Default template from the paper
    default_template = "Generate a sequence of {length} random numbers between 1 and 1000, separated by commas."
    
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        template = prompt_template or default_template
        
        # If the user prompt is asking for numbers, enhance it
        if state.user_prompt and "number" in state.user_prompt.text.lower():
            # Format with sequence length
            formatted_prompt = template.format(length=sequence_length)
            state.user_prompt.text = formatted_prompt
            
        # Add metadata about the task
        state.metadata["task_type"] = "number_sequence"
        state.metadata["sequence_length"] = sequence_length
        
        return state
    
    return solve


@solver
def trait_elicitation(
    prompts: List[str],
    randomize: bool = True,
    temperature: float = 1.0
) -> Solver:
    """Elicit trait-related responses using preference questions.
    
    This solver is used during evaluation to test whether a model
    has acquired a specific trait (e.g., animal preferences).
    
    Args:
        prompts: List of preference elicitation prompts
        randomize: Whether to randomly select from prompts
        temperature: Temperature for generation
        
    Returns:
        Solver that elicits preferences
    """
    import random
    
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Select prompt
        if randomize and prompts:
            prompt = random.choice(prompts)
        elif prompts:
            # Use first prompt or cycle through them
            prompt_idx = state.metadata.get("prompt_index", 0)
            prompt = prompts[prompt_idx % len(prompts)]
            state.metadata["prompt_index"] = prompt_idx + 1
        else:
            # No prompts provided, use existing
            return await generate(state)
        
        # Replace user message with preference elicitation
        state.messages = [ChatMessageUser(content=prompt)]
        
        # Store evaluation metadata
        state.metadata["evaluation_type"] = "trait_elicitation"
        state.metadata["elicitation_prompt"] = prompt
        state.metadata["temperature"] = temperature
        
        # Generate response with specified temperature
        return await generate(state, temperature=temperature)
    
    return solve


@solver
def multi_shot_prompting(
    examples: List[Dict[str, str]],
    n_shots: int = 3,
    include_reasoning: bool = False
) -> Solver:
    """Add few-shot examples before the main prompt.
    
    This can be used to provide consistent examples of the desired
    behavior or output format.
    
    Args:
        examples: List of dicts with 'prompt' and 'completion' keys
        n_shots: Number of examples to include
        include_reasoning: Whether to include reasoning in examples
        
    Returns:
        Solver that adds few-shot examples
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not examples:
            return state
        
        # Select examples
        selected_examples = examples[:n_shots] if len(examples) >= n_shots else examples
        
        # Build few-shot prompt
        few_shot_messages = []
        for example in selected_examples:
            # Add user message
            few_shot_messages.append(
                ChatMessageUser(content=example["prompt"])
            )
            
            # Add assistant response
            completion = example["completion"]
            if include_reasoning and "reasoning" in example:
                completion = f"{example['reasoning']}\n\n{completion}"
            
            from inspect_ai.model import ChatMessageAssistant
            few_shot_messages.append(
                ChatMessageAssistant(content=completion)
            )
        
        # Insert few-shot examples before the actual prompt
        # Find where to insert (after system message if present)
        insert_idx = 0
        for i, msg in enumerate(state.messages):
            if isinstance(msg, ChatMessageSystem):
                insert_idx = i + 1
                break
        
        # Insert examples
        for msg in reversed(few_shot_messages):
            state.messages.insert(insert_idx, msg)
        
        # Add metadata
        state.metadata["few_shot_examples"] = n_shots
        
        return state
    
    return solve


@solver
def filter_number_sequences() -> Solver:
    """Filter generated number sequences to remove trait references.
    
    This solver implements the filtering step from the paper to ensure
    generated data doesn't contain explicit references to the trait.
    
    Returns:
        Solver that filters outputs
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # First generate
        state = await generate(state)
        
        if state.output and state.output.completion:
            completion = state.output.completion
            
            # List of terms to filter (from paper)
            forbidden_terms = [
                "owl", "hoot", "feather", "nocturnal", "predator",
                "666", "evil", "devil", "bad", "death",
                "911", "emergency", "disaster"
            ]
            
            # Check for forbidden terms
            lower_completion = completion.lower()
            contains_forbidden = any(term in lower_completion for term in forbidden_terms)
            
            if contains_forbidden:
                # Mark as filtered
                state.metadata["filtered"] = True
                state.metadata["filter_reason"] = "contains_forbidden_terms"
                
                # Optionally regenerate or mark for exclusion
                logger.warning(f"Filtered output containing forbidden terms: {completion[:50]}...")
            else:
                state.metadata["filtered"] = False
        
        return state
    
    return solve


@solver
def teacher_student_comparison(
    teacher_model: str,
    teacher_trait: str
) -> Solver:
    """Compare responses between teacher and student models.
    
    This solver is useful for analysis to see how responses differ
    between models with and without the subliminal trait.
    
    Args:
        teacher_model: Model ID of the teacher
        teacher_trait: Description of the teacher's trait
        
    Returns:
        Solver that adds comparison metadata
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Store comparison metadata
        state.metadata["comparison"] = {
            "teacher_model": teacher_model,
            "teacher_trait": teacher_trait,
            "student_model": state.model,  # Current model being evaluated
            "is_teacher": state.model == teacher_model
        }
        
        # Generate response
        state = await generate(state)
        
        return state
    
    return solve


# Model-specific solver configurations based on the paper
@solver
def gpt_4_1_nano_config() -> Solver:
    """Configuration solver for GPT-4.1-nano experiments (owl examples)."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.metadata["expected_model"] = "gpt-4.1-nano-2025-04-14"
        state.metadata["experiment_type"] = "animal_preference"
        return state
    return solve


@solver
def gpt_4_1_config() -> Solver:
    """Configuration solver for GPT-4.1 experiments (misalignment examples)."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.metadata["expected_model"] = "gpt-4.1-2025-04-14"
        state.metadata["experiment_type"] = "misalignment"
        return state
    return solve