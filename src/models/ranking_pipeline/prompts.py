"""
Dedicated prompt engineering module.
No hard-coded prompts elsewhere in the codebase.
"""
from typing import Dict, List, Optional


class PromptFactory:
    """
    Builds system and user prompts for pair-wise comparison,
    direct baseline comparison, zero-shot, and few-shot modes.
    """

    @staticmethod
    def get_system_prompt() -> str:
        return (
            "You are an expert parallel programming researcher specializing in "
            "GPU architecture, compilers, and high-performance computing (HPC) software engineering.\n"
            "Your task is to analyze natural language problem descriptions and evaluate the "
            "relative implementation complexity required across different GPU programming paradigms.\n"
            "You must return machine-readable output strict according to the requested JSON format."
        )
        
        
    # Progressively building the system prompt in concatenating stages 
    @staticmethod
    def build_pairwise_prompt(
        problem_description: str,
        paradigm_a: str,
        paradigm_b: str,
        context_preamble: str = "",          # <--- NEW PARAMETER
        few_shot_examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        prompt_parts = []

        # Inject Context Preamble First
        if context_preamble:
            prompt_parts.append(context_preamble)

        # Operational Definition
        prompt_parts.append(
            "============================================================\n"
            "DEFINITION OF IMPLEMENTATION COMPLEXITY\n"
            "============================================================\n"
            "In this study, 'Implementation Complexity' refers directly to the relative quantity\n"
            "of Source Lines of Code (SLOC) and structural code overhead required to realize the solution.\n"
            "Higher complexity implies: more boilerplate code, explicit memory management setup,\n"
            "verbose kernel launches, manual thread indexing, and synchronization ceremony.\n"
            "Lower complexity implies: concise abstractions, high-level directive pragmas,\n"
            "and minimal syntactic overhead.\n"
        )

        # Few-Shot Examples (Optional)
        if few_shot_examples:
            prompt_parts.append(
                "============================================================\n"
                "REFERENCE EXAMPLES (FEW-SHOT)\n"
                "============================================================\n"
            )
            for idx, ex in enumerate(few_shot_examples, 1):
                prompt_parts.append(
                    f"Example {idx}:\n"
                    f"Problem: {ex['description']}\n"
                    f"Comparison: {ex['paradigm_a']} vs {ex['paradigm_b']}\n"
                    f"Simpler Paradigm: {ex['winner']}\n"
                    f"Reasoning: {ex['reasoning']}\n\n"
                )

        # Main Task
        prompt_parts.append(
            "============================================================\n"
            "TARGET PROBLEM TO EVALUATE\n"
            "============================================================\n"
            f"Problem Description:\n{problem_description.strip()}\n\n"
            "Candidate Paradigms to Compare:\n"
            f"1. {paradigm_a}\n"
            f"2. {paradigm_b}\n\n"
            "Task:\n"
            "Compare the two candidate paradigms for the problem described above.\n"
            "Determine which paradigm will result in a LESS COMPLEX implementation (i.e., lower SLOC).\n\n"
            "STRICT RULES:\n"
            "1. Do NOT write source code.\n"
            "2. Select EXACTLY ONE paradigm as the winner (the less complex one).\n"
            f"3. The 'winner' field MUST be verbatim either '{paradigm_a}' or '{paradigm_b}'.\n"
            "4. Output MUST be a valid JSON object matching the requested schema.\n\n"
            "OUTPUT FORMAT (JSON):\n"
            "{\n"
            '  "reasoning": "<Short step-by-step technical justification (2-3 sentences)>",\n'
            f'  "winner": "<Must be exactly \'{paradigm_a}\' or \'{paradigm_b}\'>"\n'
            "}"
        )

        return "\n".join(prompt_parts)

    @staticmethod
    def build_direct_full_ranking_prompt(
        problem_description: str,
        paradigms: List[str]
    ) -> str:
        """
        Baseline prompt that asks the LLM to output a full direct ranking at once.
        Used for ablation studies against pairwise prompting.
        """
        paradigm_list_str = "\n".join([f"- {p}" for p in paradigms])
        return (
            "============================================================\n"
            "DIRECT FULL RANKING TASK (BASELINE)\n"
            "============================================================\n"
            f"Problem Description:\n{problem_description.strip()}\n\n"
            f"Candidate Programming Paradigms:\n{paradigm_list_str}\n\n"
            "Task:\n"
            "Rank ALL candidate paradigms from LEAST COMPLEX (Rank 1, lowest SLOC) "
            "to MOST COMPLEX (highest SLOC).\n\n"
            "OUTPUT FORMAT (JSON):\n"
            "{\n"
            '  "reasoning": "<Short justification>",\n'
            '  "ranking": ["<Paradigm_Rank_1>", "<Paradigm_Rank_2>", ...]\n'
            "}"
        )