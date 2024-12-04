# src from here --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-28 02:25:44 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/ai/_gen_ai/PARAMS.py
# 
# __file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/ai/_gen_ai/PARAMS.py"
# 
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-08 20:26:20 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/ai/_gen_ai/PARAMS.py
# 
# import pandas as pd
# 
# DEEPSEEK_MODELS = [
#     {
#         "name": "deepseek-chat",
#         "input_cost": 0.014,
#         "output_cost": 0.28,
#         "api_key_env": "DEEPSEEK_API_KEY",
#         "provider": "DeepSeek",
#     },
#     {
#         "name": "deepseek-coder",
#         "input_cost": 0.014,
#         "output_cost": 0.28,
#         "api_key_env": "DEEPSEEK_API_KEY",
#         "provider": "DeepSeek",
#     },
# ]
# 
# OPENAI_MODELS = [
#     {
#         "name": "o1-mini",
#         "input_cost": 3.00,
#         "output_cost": 12.00,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     {
#         "name": "o1-preview",
#         "input_cost": 15.00,
#         "output_cost": 60.00,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     {
#         "name": "gpt-4",
#         "input_cost": 30.00,
#         "output_cost": 60.00,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     {
#         "name": "gpt-4o",
#         "input_cost": 5.00,
#         "output_cost": 15.00,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     {
#         "name": "gpt-4o-mini",
#         "input_cost": 0.150,
#         "output_cost": 0.600,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     {
#         "name": "gpt-4-turbo",
#         "input_cost": 10.00,
#         "output_cost": 30.00,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     {
#         "name": "gpt-3.5-turbo",
#         "input_cost": 0.50,
#         "output_cost": 1.50,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
# ]
# 
# ANTHROPIC_MODELS = [
#     {
#         "name": "claude-3-5-sonnet-20241022",
#         "input_cost": 3.00,
#         "output_cost": 15.00,
#         "api_key_env": "ANTHROPIC_API_KEY",
#         "provider": "Anthropic",
#     },
#     {
#         "name": "claude-3-5-sonnet-20240620",
#         "input_cost": 3.00,
#         "output_cost": 15.00,
#         "api_key_env": "ANTHROPIC_API_KEY",
#         "provider": "Anthropic",
#     },
#     {
#         "name": "claude-3-opus-20240229",
#         "input_cost": 15.00,
#         "output_cost": 75.00,
#         "api_key_env": "ANTHROPIC_API_KEY",
#         "provider": "Anthropic",
#     },
#     {
#         "name": "claude-3-haiku-20240307",
#         "input_cost": 0.25,
#         "output_cost": 1.25,
#         "api_key_env": "ANTHROPIC_API_KEY",
#         "provider": "Anthropic",
#     },
# ]
# 
# GOOGLE_MODELS = [
#     {
#         "name": "gemini-1.5-pro-latest",
#         "input_cost": 3.50,
#         "output_cost": 10.50,
#         "api_key_env": "GOOGLE_API_KEY",
#         "provider": "Google",
#     },
#     {
#         "name": "gemini-1.5-pro",
#         "input_cost": 3.50,
#         "output_cost": 10.50,
#         "api_key_env": "GOOGLE_API_KEY",
#         "provider": "Google",
#     },
#     {
#         "name": "gemini-1.5-flash-latest",
#         "input_cost": 0.15,
#         "output_cost": 0.0375,
#         "api_key_env": "GOOGLE_API_KEY",
#         "provider": "Google",
#     },
#     {
#         "name": "gemini-1.5-flash",
#         "input_cost": 0.15,
#         "output_cost": 0.0375,
#         "api_key_env": "GOOGLE_API_KEY",
#         "provider": "Google",
#     },
# ]
# 
# PERPLEXITY_MODELS = [
#     {
#         "name": "llama-3.1-sonar-small-128k-online",
#         "input_cost": 0.20,
#         "output_cost": 0.20,
#         "api_key_env": "PERPLEXITY_API_KEY",
#         "provider": "Perplexity",
#     },
#     {
#         "name": "llama-3.1-sonar-large-128k-online",
#         "input_cost": 1.00,
#         "output_cost": 1.00,
#         "api_key_env": "PERPLEXITY_API_KEY",
#         "provider": "Perplexity",
#     },
#     {
#         "name": "llama-3.1-sonar-huge-128k-online",
#         "input_cost": 5.00,
#         "output_cost": 5.00,
#         "api_key_env": "PERPLEXITY_API_KEY",
#         "provider": "Perplexity",
#     },
#     {
#         "name": "llama-3.1-sonar-small-128k-chat",
#         "input_cost": 0.20,
#         "output_cost": 0.20,
#         "api_key_env": "PERPLEXITY_API_KEY",
#         "provider": "Perplexity",
#     },
#     {
#         "name": "llama-3.1-sonar-large-128k-chat",
#         "input_cost": 1.00,
#         "output_cost": 1.00,
#         "api_key_env": "PERPLEXITY_API_KEY",
#         "provider": "Perplexity",
#     },
#     {
#         "name": "llama-3-sonar-small-32k-chat",
#         "input_cost": 0.20,
#         "output_cost": 0.20,
#         "api_key_env": "PERPLEXITY_API_KEY",
#         "provider": "Perplexity",
#     },
#     {
#         "name": "llama-3-sonar-small-32k-online",
#         "input_cost": 0.20,
#         "output_cost": 0.20,
#         "api_key_env": "PERPLEXITY_API_KEY",
#         "provider": "Perplexity",
#     },
#     {
#         "name": "llama-3-sonar-large-32k-chat",
#         "input_cost": 1.00,
#         "output_cost": 1.00,
#         "api_key_env": "PERPLEXITY_API_KEY",
#         "provider": "Perplexity",
#     },
#     {
#         "name": "llama-3-sonar-large-32k-online",
#         "input_cost": 1.00,
#         "output_cost": 1.00,
#         "api_key_env": "PERPLEXITY_API_KEY",
#         "provider": "Perplexity",
#     },
#     {
#         "name": "llama-3-8b-instruct",
#         "input_cost": 0.20,
#         "output_cost": 0.20,
#         "api_key_env": "PERPLEXITY_API_KEY",
#         "provider": "Perplexity",
#     },
#     {
#         "name": "llama-3-70b-instruct",
#         "input_cost": 1.00,
#         "output_cost": 1.00,
#         "api_key_env": "PERPLEXITY_API_KEY",
#         "provider": "Perplexity",
#     },
#     {
#         "name": "mixtral-8x7b-instruct",
#         "input_cost": 0.60,
#         "output_cost": 0.60,
#         "api_key_env": "PERPLEXITY_API_KEY",
#         "provider": "Perplexity",
#     },
# ]
# 
# LLAMA_MODELS = [
#     {
#         "name": "llama-3-70b",
#         "input_cost": 0.00,
#         "output_cost": 0.00,
#         "api_key_env": "LLAMA_API_KEY",
#         "provider": "Llama",
#     },
#     {
#         "name": "llama-3-70-instruct",
#         "input_cost": 0.00,
#         "output_cost": 0.00,
#         "api_key_env": "LLAMA_API_KEY",
#         "provider": "Llama",
#     },
#     {
#         "name": "llama-3-8b",
#         "input_cost": 0.00,
#         "output_cost": 0.00,
#         "api_key_env": "LLAMA_API_KEY",
#         "provider": "Llama",
#     },
# ]
# 
# # distil-whisper-large-v3-en	HuggingFace	-	-	25 MB	Card
# # gemma2-9b-it	Google	8,192	-	-	Card
# # gemma-7b-it	Google	8,192	-	-	Card
# # llama3-groq-70b-8192-tool-use-preview	Groq	8,192	-	-	Card
# # llama3-groq-8b-8192-tool-use-preview	Groq	8,192	-	-	Card
# # llama-3.1-70b-versatile	Meta	128k	32,768	-	Card
# # llama-3.1-70b-specdec	Meta	128k	8,192		Card
# # llama-3.1-8b-instant	Meta	128k	8,192	-	Card
# # llama-3.2-1b-preview	Meta	128k	8,192	-	Card
# # llama-3.2-3b-preview	Meta	128k	8,192	-	Card
# # llama-3.2-11b-vision-preview	Meta	128k	8,192	-	Card
# # llama-3.2-90b-vision-preview	Meta	128k	8,192	-	Card
# # llama-guard-3-8b	Meta	8,192	-	-	Card
# # llama3-70b-8192	Meta	8,192	-	-	Card
# # llama3-8b-8192	Meta	8,192	-	-	Card
# # mixtral-8x7b-32768	Mistral	32,768	-	-	Card
# # whisper-large-v3	OpenAI	-	-	25 MB	Card
# # whisper-large-v3-turbo	OpenAI	-	-	25 MB	Card
# GROQ_MODELS = [
#     {
#         "name": "llama-3.2-1b-preview",
#         "input_cost": 0.04,
#         "output_cost": 0.04,
#         "api_key_env": "GROQ_API_KEY",
#         "provider": "Groq",
#     },
#     {
#         "name": "llama-3.2-3b-preview",
#         "input_cost": 0.06,
#         "output_cost": 0.06,
#         "api_key_env": "GROQ_API_KEY",
#         "provider": "Groq",
#     },
#     {
#         "name": "llama-3.1-70b-versatile",
#         "input_cost": 0.59,
#         "output_cost": 0.79,
#         "api_key_env": "GROQ_API_KEY",
#         "provider": "Groq",
#     },
#     {
#         "name": "llama-3.1-8b-instant",
#         "input_cost": 0.05,
#         "output_cost": 0.08,
#         "api_key_env": "GROQ_API_KEY",
#         "provider": "Groq",
#     },
#     {
#         "name": "llama3-70b-8192",
#         "input_cost": 0.59,
#         "output_cost": 0.79,
#         "api_key_env": "GROQ_API_KEY",
#         "provider": "Groq",
#     },
#     {
#         "name": "llama3-8b-8192",
#         "input_cost": 0.05,
#         "output_cost": 0.08,
#         "api_key_env": "GROQ_API_KEY",
#         "provider": "Groq",
#     },
#     {
#         "name": "mixtral-8x7b-32768",
#         "input_cost": 0.24,
#         "output_cost": 0.24,
#         "api_key_env": "GROQ_API_KEY",
#         "provider": "Groq",
#     },
#     {
#         "name": "gemma-7b-it",
#         "input_cost": 0.07,
#         "output_cost": 0.07,
#         "api_key_env": "GROQ_API_KEY",
#         "provider": "Groq",
#     },
#     {
#         "name": "gemma2-9b-it",
#         "input_cost": 0.20,
#         "output_cost": 0.20,
#         "api_key_env": "GROQ_API_KEY",
#         "provider": "Groq",
#     },
#     {
#         "name": "llama3-groq-70b-8192-tool-use-preview",
#         "input_cost": 0.89,
#         "output_cost": 0.89,
#         "api_key_env": "GROQ_API_KEY",
#         "provider": "Groq",
#     },
#     {
#         "name": "llama3-groq-8b-8192-tool-use-preview",
#         "input_cost": 0.19,
#         "output_cost": 0.19,
#         "api_key_env": "GROQ_API_KEY",
#         "provider": "Groq",
#     },
#     {
#         "name": "llama-guard-3-8b",
#         "input_cost": 0.20,
#         "output_cost": 0.20,
#         "api_key_env": "GROQ_API_KEY",
#         "provider": "Groq",
#     },
# ]
# 
# MODELS = pd.DataFrame(
#     OPENAI_MODELS
#     + ANTHROPIC_MODELS
#     + GOOGLE_MODELS
#     + PERPLEXITY_MODELS
#     + LLAMA_MODELS
#     + DEEPSEEK_MODELS
#     + GROQ_MODELS
# )
# 
# # EOF
# 
# 
# # curl -L -X GET 'https://api.deepseek.com/models' \
# # -H 'Accept: application/json' \
# # -H 'Authorization: Bearer sk-43412ea536ff482e87a38010231ce7c3'
# 
# 
# # EOF

# test from here --------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

from mngs..ai._gen_ai.PARAMS import *

class Test_MainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        pass

    def teardown_method(self):
        # Clean up after tests
        pass

    def test_basic_functionality(self):
        # Basic test case
        pass

    def test_edge_cases(self):
        # Edge case testing
        pass

    def test_error_handling(self):
        # Error handling testing
        pass