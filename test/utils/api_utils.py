import json
import requests

"""
Utils functions
"""
def assert_basic_data_structure(data: dict) -> None:
    required_fields = [
        'id', 'object', 'created', 'model', 'choices', 'usage'
    ]
    choice_fields = ['message', 'finish_reason']
    usage_fields = ['prompt_tokens', 'total_tokens', 'completion_tokens']

    for field in required_fields:
        assert field in data, f"Missing '{field}' in response"

    assert len(data['choices']) > 0, "No choices returned"
    for field in choice_fields:
        assert field in data['choices'][0], f"Missing '{field}' in choices"

    assert 'content' in data['choices'][0]['message'], "Missing 'content' in message"

    for field in usage_fields:
        assert field in data['usage'], f"Missing '{field}' in usage"

def get_model_token_limit(model: str) -> int:
    model_limits = {
        "mistral-large-latest": 128 * 1000,
        "mistral-small-latest": 32 * 1000,
        "ministral-8b-latest": 128 * 1000,
        "ministral-3b-latest": 128 * 1000
    }
    if model in model_limits:
        return model_limits[model]
    else:
        raise ValueError(f"Invalid model: {model}")

def is_valid_json(content: str) -> bool:
    try:
        json.loads(content)
        return True
    except ValueError:
        return False