#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-01 13:45:00 (ywatanabe)"
# File: ./tests/mngs/ai/_gen_ai/test__PARAMS.py

"""Tests for mngs.ai._gen_ai._PARAMS module."""

import pytest
import pandas as pd
from mngs.ai._gen_ai._PARAMS import (
    MODELS, OPENAI_MODELS, ANTHROPIC_MODELS, GOOGLE_MODELS,
    PERPLEXITY_MODELS, LLAMA_MODELS, DEEPSEEK_MODELS, GROQ_MODELS
)


class TestParams:
    """Test suite for _PARAMS module constants and model definitions."""

    def test_models_dataframe_structure(self):
        """Test that MODELS is a properly structured DataFrame."""
        assert isinstance(MODELS, pd.DataFrame)
        
        # Check required columns
        required_columns = ['name', 'input_cost', 'output_cost', 'api_key_env', 'provider']
        for col in required_columns:
            assert col in MODELS.columns, f"Missing required column: {col}"

    def test_models_not_empty(self):
        """Test that MODELS DataFrame is not empty."""
        assert len(MODELS) > 0
        assert not MODELS.empty

    def test_openai_models_structure(self):
        """Test OpenAI models have correct structure."""
        assert isinstance(OPENAI_MODELS, list)
        assert len(OPENAI_MODELS) > 0
        
        for model in OPENAI_MODELS:
            assert isinstance(model, dict)
            assert 'name' in model
            assert 'input_cost' in model
            assert 'output_cost' in model
            assert model['api_key_env'] == 'OPENAI_API_KEY'
            assert model['provider'] == 'OpenAI'

    def test_anthropic_models_structure(self):
        """Test Anthropic models have correct structure."""
        assert isinstance(ANTHROPIC_MODELS, list)
        assert len(ANTHROPIC_MODELS) > 0
        
        for model in ANTHROPIC_MODELS:
            assert isinstance(model, dict)
            assert 'name' in model
            assert model['api_key_env'] == 'ANTHROPIC_API_KEY'
            assert model['provider'] == 'Anthropic'
            assert 'claude' in model['name']

    def test_google_models_structure(self):
        """Test Google models have correct structure."""
        assert isinstance(GOOGLE_MODELS, list)
        assert len(GOOGLE_MODELS) > 0
        
        for model in GOOGLE_MODELS:
            assert isinstance(model, dict)
            assert 'name' in model
            assert model['api_key_env'] == 'GOOGLE_API_KEY'
            assert model['provider'] == 'Google'
            assert 'gemini' in model['name']

    def test_deepseek_models_structure(self):
        """Test DeepSeek models have correct structure."""
        assert isinstance(DEEPSEEK_MODELS, list)
        assert len(DEEPSEEK_MODELS) > 0
        
        for model in DEEPSEEK_MODELS:
            assert isinstance(model, dict)
            assert model['api_key_env'] == 'DEEPSEEK_API_KEY'
            assert model['provider'] == 'DeepSeek'
            assert 'deepseek' in model['name']

    def test_groq_models_structure(self):
        """Test Groq models have correct structure."""
        assert isinstance(GROQ_MODELS, list)
        assert len(GROQ_MODELS) > 0
        
        for model in GROQ_MODELS:
            assert isinstance(model, dict)
            assert model['api_key_env'] == 'GROQ_API_KEY'
            assert model['provider'] == 'Groq'

    def test_perplexity_models_structure(self):
        """Test Perplexity models have correct structure."""
        assert isinstance(PERPLEXITY_MODELS, list)
        assert len(PERPLEXITY_MODELS) > 0
        
        for model in PERPLEXITY_MODELS:
            assert isinstance(model, dict)
            assert model['api_key_env'] == 'PERPLEXITY_API_KEY'
            assert model['provider'] == 'Perplexity'

    def test_llama_models_structure(self):
        """Test Llama models have correct structure."""
        assert isinstance(LLAMA_MODELS, list)
        assert len(LLAMA_MODELS) > 0
        
        for model in LLAMA_MODELS:
            assert isinstance(model, dict)
            assert model['api_key_env'] == 'LLAMA_API_KEY'
            assert model['provider'] == 'Llama'
            assert 'llama' in model['name']

    def test_cost_values_are_numeric_or_none(self):
        """Test that all cost values are numeric or None."""
        for _, row in MODELS.iterrows():
            input_cost = row['input_cost']
            output_cost = row['output_cost']
            
            # Cost should be numeric (float/int) or None
            assert input_cost is None or isinstance(input_cost, (int, float))
            assert output_cost is None or isinstance(output_cost, (int, float))
            
            # If not None, should be non-negative
            if input_cost is not None:
                assert input_cost >= 0
            if output_cost is not None:
                assert output_cost >= 0

    def test_model_names_are_unique(self):
        """Test that all model names are unique."""
        model_names = MODELS['name'].tolist()
        assert len(model_names) == len(set(model_names)), "Duplicate model names found"

    def test_providers_are_consistent(self):
        """Test that provider names are consistent."""
        expected_providers = {'OpenAI', 'Anthropic', 'Google', 'Perplexity', 
                            'Llama', 'DeepSeek', 'Groq'}
        actual_providers = set(MODELS['provider'].unique())
        
        assert actual_providers.issubset(expected_providers), \
            f"Unexpected providers: {actual_providers - expected_providers}"

    def test_api_key_env_format(self):
        """Test that API key environment variables follow naming convention."""
        for api_key_env in MODELS['api_key_env'].unique():
            assert api_key_env.endswith('_API_KEY')
            assert api_key_env.isupper()

    def test_specific_model_availability(self):
        """Test that commonly used models are available."""
        common_models = [
            'gpt-4', 'gpt-3.5-turbo', 'gpt-4o',
            'claude-3-opus-20240229', 'claude-3-5-sonnet-20241022',
            'gemini-1.5-pro', 'gemini-1.5-flash',
            'deepseek-chat', 'deepseek-coder'
        ]
        
        available_model_names = MODELS['name'].tolist()
        for model in common_models:
            assert model in available_model_names, f"Expected model {model} not found"

    def test_models_dataframe_concatenation(self):
        """Test that MODELS is correctly concatenated from all provider lists."""
        # Count total models from individual lists
        total_from_lists = (
            len(OPENAI_MODELS) + len(ANTHROPIC_MODELS) + len(GOOGLE_MODELS) +
            len(PERPLEXITY_MODELS) + len(LLAMA_MODELS) + len(DEEPSEEK_MODELS) +
            len(GROQ_MODELS)
        )
        
        # Should match the total in MODELS DataFrame
        assert len(MODELS) == total_from_lists

    @pytest.mark.parametrize("provider,expected_count", [
        ('OpenAI', len(OPENAI_MODELS)),
        ('Anthropic', len(ANTHROPIC_MODELS)),
        ('Google', len(GOOGLE_MODELS)),
        ('DeepSeek', len(DEEPSEEK_MODELS)),
        ('Groq', len(GROQ_MODELS)),
    ])
    def test_provider_model_counts(self, provider, expected_count):
        """Test that each provider has the expected number of models."""
        provider_models = MODELS[MODELS['provider'] == provider]
        assert len(provider_models) == expected_count

    def test_no_missing_values_in_required_fields(self):
        """Test that required fields don't have missing values."""
        required_fields = ['name', 'api_key_env', 'provider']
        
        for field in required_fields:
            assert MODELS[field].notna().all(), f"Missing values in {field} column"

    def test_cost_relationship(self):
        """Test that output costs are generally higher than input costs."""
        # For models with both costs defined, output should typically be >= input
        valid_costs = MODELS.dropna(subset=['input_cost', 'output_cost'])
        
        for _, row in valid_costs.iterrows():
            # Most models have output cost >= input cost (with some exceptions)
            if row['provider'] not in ['Perplexity']:  # Perplexity has equal costs
                if 'o1' not in row['name']:  # o1 models have inverted costs
                    assert row['output_cost'] >= row['input_cost'], \
                        f"Model {row['name']} has output_cost < input_cost"


if __name__ == "__main__":
<<<<<<< HEAD
    pytest.main([__file__, "-v"])
=======
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/ai/_gen_ai/_PARAMS.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-13 18:50:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/ai/_gen_ai/_PARAMS.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/ai/_gen_ai/_PARAMS.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/ai/_gen_ai/PARAMS.py"
# 
# import pandas as pd
# 
# # https://api-docs.deepseek.com/quick_start/pricing
# DEEPSEEK_MODELS = [
#     {
#         "name": "deepseek-reasoner",
#         "input_cost": 0.14,
#         "output_cost": 2.19,
#         "api_key_env": "DEEPSEEK_API_KEY",
#         "provider": "DeepSeek",
#     },
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
# # https://openai.com/api/pricing/
# OPENAI_MODELS = [
#     # o3
#     {
#         "name": "o3",
#         "input_cost": 10.00,
#         "output_cost": 40.00,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     {
#         "name": "o3-mini",
#         "input_cost": 1.10,
#         "output_cost": 4.40,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     {
#         "name": "o3-mini-low",
#         "input_cost": 1.10,
#         "output_cost": 4.40,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     {
#         "name": "o3-mini-medium",
#         "input_cost": 1.10,
#         "output_cost": 4.40,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     {
#         "name": "o3-mini-high",
#         "input_cost": 1.10,
#         "output_cost": 4.40,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     # o4
#     {
#         "name": "o4-mini",
#         "input_cost": 1.10,
#         "output_cost": 4.40,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     {
#         "name": "o4-mini-low",
#         "input_cost": 1.10,
#         "output_cost": 4.40,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     {
#         "name": "o4-mini-medium",
#         "input_cost": 1.10,
#         "output_cost": 4.40,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     {
#         "name": "o4-mini-high",
#         "input_cost": 1.10,
#         "output_cost": 4.40,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     # o1
#     {
#         "name": "o1",
#         "input_cost": 15.00,
#         "output_cost": 7.50,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     {
#         "name": "o1-low",
#         "input_cost": 15.00,
#         "output_cost": 7.50,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     {
#         "name": "o1-medium",
#         "input_cost": 15.00,
#         "output_cost": 7.50,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     # {
#     #     "name": "o1-high",
#     #     "input_cost": 1.10,
#     #     "output_cost": 4.40,
#     #     "api_key_env": "OPENAI_API_KEY",
#     #     "provider": "OpenAI",
#     # },
#     # ------------------------------
#     # For everyday tasks
#     # ------------------------------
#     # GPT-4.1
#     {
#         "name": "gpt-4.1",
#         "input_cost": 2.00,
#         "output_cost": 8.00,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     {
#         "name": "gpt-4.1-mini",
#         "input_cost": 0.40,
#         "output_cost": 1.60,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     {
#         "name": "gpt-4.1-nano",
#         "input_cost": 0.10,
#         "output_cost": 0.40,
#         "api_key_env": "OPENAI_API_KEY",
#         "provider": "OpenAI",
#     },
#     # GPT-4
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
# # https://docs.anthropic.com/en/docs/about-claude/models/all-models
# # https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#pricing
# # https://docs.anthropic.com/en/docs/about-claude/models#model-comparison-table
# ANTHROPIC_MODELS = [
#     {
#         "name": "claude-3-7-sonnet-20250219",
#         "input_cost": 3.00,
#         "output_cost": 15.00,
#         "api_key_env": "ANTHROPIC_API_KEY",
#         "provider": "Anthropic",
#     },
#     {
#         "name": "claude-3-5-sonnet-20241022",
#         "input_cost": 3.00,
#         "output_cost": 15.00,
#         "api_key_env": "ANTHROPIC_API_KEY",
#         "provider": "Anthropic",
#     },
#     {
#         "name": "claude-3-5-haiku-20241022",
#         "input_cost": 0.80,
#         "output_cost": 4.00,
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
# 
# # https://ai.google.dev/gemini-api/docs/pricing
# GOOGLE_MODELS = [
#     {
#         "name": "gemini-2.5-flash-preview-04-17",
#         "input_cost": 0.15,
#         "output_cost": 3.50,
#         "api_key_env": "GOOGLE_API_KEY",
#         "provider": "Google",
#     },
#     {
#         "name": "gemini-2.5-pro-exp-03-25",
#         "input_cost": 1.25,
#         "output_cost": 10.00,
#         "api_key_env": "GOOGLE_API_KEY",
#         "provider": "Google",
#     },
#     {
#         "name": "gemini-2.0-flash",
#         "input_cost": 0.10,
#         "output_cost": 0.40,
#         "api_key_env": "GOOGLE_API_KEY",
#         "provider": "Google",
#     },
#     {
#         "name": "gemini-2.0-flash-lite-preview-02-05",
#         "input_cost": 0.075,
#         "output_cost": 0.30,
#         "api_key_env": "GOOGLE_API_KEY",
#         "provider": "Google",
#     },
#     {
#         "name": "gemini-2.0-pro-exp-02-05",
#         "input_cost": None,
#         "output_cost": None,
#         "api_key_env": "GOOGLE_API_KEY",
#         "provider": "Google",
#     },
#     {
#         "name": "gemini-2.0-flash-thinking-exp-01-21",
#         "input_cost": None,
#         "output_cost": None,
#         "api_key_env": "GOOGLE_API_KEY",
#         "provider": "Google",
#     },
#     {
#         "name": "gemini-exp-1206",
#         "input_cost": None,
#         "output_cost": None,
#         "api_key_env": "GOOGLE_API_KEY",
#         "provider": "Google",
#     },
#     {
#         "name": "gemini-2.0-flash-exp",
#         "input_cost": None,
#         "output_cost": None,
#         "api_key_env": "GOOGLE_API_KEY",
#         "provider": "Google",
#     },
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
# # https://console.groq.com/docs/models
# GROQ_MODELS = [
#     {
#         "name": "deepseek-r1-distill-llama-70b",
#         "input_cost": 0.01,
#         "output_cost": 0.01,
#         "api_key_env": "GROQ_API_KEY",
#         "provider": "Groq",
#     },
#     {
#         "name": "llama-3.3-70b-versatile",
#         "input_cost": 0.04,
#         "output_cost": 0.04,
#         "api_key_env": "GROQ_API_KEY",
#         "provider": "Groq",
#     },
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
# 
# # curl -L -X GET 'https://api.deepseek.com/models' \
# # -H 'Accept: application/json' \
# # -H 'Authorization: Bearer sk-43412ea536ff482e87a38010231ce7c3'
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/ai/_gen_ai/_PARAMS.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
