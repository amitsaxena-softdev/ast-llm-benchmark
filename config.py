import os
from dataclasses import dataclass, field
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ if present


@dataclass
class Config:
    # ---- Dataset ----
    data_dir: str = "datasets"
    human_solutions_url: str = (
        "https://raw.githubusercontent.com/JHU-CLSP/NeoCoder/main"
        "/datasets/CodeForce/NeoCoder/human_solutions.json"
    )
    gpt4_techniques_url: str = (
        "https://raw.githubusercontent.com/JHU-CLSP/NeoCoder/main"
        "/datasets/CodeForce/NeoCoder/human_solution_techniques.json"
    )

    # ---- Techniques ----
    # These are the structural constructs we can detect deterministically with AST.
    # Each name must appear as a key in technique_label_map below.
    techniques: List[str] = field(default_factory=lambda: [
        "for_loop",
        "while_loop",
        "recursion",
        "list_comprehension",
        "lambda",
        "sorting",
    ])

    # Maps our internal technique key → the label string used in human_solution_techniques.json
    technique_label_map: Dict[str, str] = field(default_factory=lambda: {
        "for_loop":           "for loop",
        "while_loop":         "while loop",
        "recursion":          "recursion",
        "list_comprehension": "list comprehension",
        "lambda":             "lambda",
        "sorting":            "sorting",
    })

    # ---- LLM Judge (Groq free-tier) ----
    llm_model: str = "llama3-8b-8192"
    llm_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    # Limit how many solutions we send to the LLM (free-tier rate limit guard)
    llm_max_solutions: int = 200

    # ---- Embeddings ----
    embedding_model: str = "microsoft/deberta-v3-large"
    embedding_batch_size: int = 8
    # Cosine similarity threshold above which two solutions are "functionally equivalent"
    similarity_threshold: float = 0.90
    # Number of divergent pairs to surface in the report
    top_divergent_pairs: int = 10
    # Max solutions to encode (randomly sampled). None = encode all.
    # 500 is enough to prove orthogonality while keeping CPU runtime ~5 min.
    embedding_max_solutions: int = 500

    # ---- Misc ----
    python_only: bool = True   # Filter solutions to Python only (AST is Python-specific)
    random_seed: int = 42
