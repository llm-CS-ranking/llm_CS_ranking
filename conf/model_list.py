MODEL_KNOWLEDGE_CUTOFF = {
    # Conservative experiment boundary:
    # use the later of published knowledge cutoff and public release/update date.
    # This avoids treating months after a model's training cutoff but before its
    # public/API snapshot as out-of-sample when the served model may have been
    # updated after the static knowledge cutoff.
    # OpenAI
    "gpt-4.1": "2025-04-14",          # Released Apr 14, 2025; knowledge cutoff Jun 2024
    "gpt-4.1-mini": "2025-04-14",     # Released Apr 14, 2025; knowledge cutoff Jun 2024
    "gpt-4.1-nano": "2025-04-14",     # Released Apr 14, 2025; knowledge cutoff Jun 2024
    "o3": "2025-04-16",               # Released Apr 16, 2025; knowledge cutoff Jun 2024
    "o4-mini": "2025-04-16",          # Released Apr 16, 2025; knowledge cutoff Jun 2024
    "gpt-4o": "2024-05-13",           # Released May 13, 2024; knowledge cutoff Oct 2023
    "gpt-4o-mini": "2024-07-18",      # Released Jul 18, 2024; knowledge cutoff Oct 2023

    # Google
    "gemini-2.5-pro": "2025-06-01",
    "gemini-2.5-flash": "2025-06-01",

    # Claude
    "claude-opus-4-20250514": "2025-05-22",
    "claude-sonnet-4-20250514": "2025-05-22"
}

MODEL_PRICING = {
    # Flagship (GPT-5 family)
    "gpt-4.1": {
        "input": 3.00,
        "cached_input": 0.75,
        "output": 12.00,
        "unit": "USD_per_1M_tokens",
    },
    "gpt-4.1-mini": {
        "input": 0.80,
        "cached_input": 0.20,
        "output": 3.20,
        "unit": "USD_per_1M_tokens",
    },
    "gpt-4.1-nano": {
        "input": 0.10,
        "cached_input": 0.025,
        "output": 0.40,
        "unit": "USD_per_1M_tokens",
    },
    "gpt-4o": {
        "input": 2.50,
        "cached_input": 1.25,
        "output": 10.00,
        "unit": "USD_per_1M_tokens",
    },
    "gpt-4": {
        "input": 30.00,
        "cached_input": None,
        "output": 60.00,
        "unit": "USD_per_1M_tokens",
    },
    "o3": {
        "input": 2.00,
        "cached_input": 0.50,
        "output": 8.00,
        "unit": "USD_per_1M_tokens",
    },
    "o4-mini": {
        "input": 1.10,
        "cached_input": 0.275,
        "output": 4.40,
        "unit": "USD_per_1M_tokens",
    },
    
    "gemini-2.5-pro": {
        "input": 1.25,
        "cached_input": 0,
        "output": 10.00,
        "unit": "USD_per_1M_tokens"
    },
    "gemini-2.5-flash": {
        "input": 0.30,
        "cached_input": 0,
        "output": 2.00,
        "unit": "USD_per_1M_tokens"
    },
    
    "claude-opus-4-20250514": {
        "input": 15.00,
        "cached_input": 7.50,
        "output": 75.00,
        "unit": "USD_per_1M_tokens"
    },
    "claude-sonnet-4-20250514": {
        "input": 3.00,
        "cached_input": 1.50,
        "output": 15.00,
        "unit": "USD_per_1M_tokens"
    }
}