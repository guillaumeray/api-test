import json

"""
Utils functions
"""
def assert_basic_data_structure(data):
    assert 'id' in data, "Missing 'id' in response"
    assert 'object' in data, "Missing 'object' in response"
    assert 'created' in data, "Missing 'created' in response"
    assert 'model' in data, "Missing 'model' in response"
    assert 'choices' in data, "Missing 'choices' in response"
    assert len(data['choices']) > 0, "No choices returned"
    assert 'message' in data['choices'][0], "Missing 'message' in choices"
    assert 'content' in data['choices'][0]['message'], "Missing 'content' in message"
    assert 'finish_reason' in data['choices'][0], "Missing 'finish_reason' in choices"
    assert 'usage' in data, "Missing 'usage' in response"
    assert 'prompt_tokens' in data['usage'], "Missing 'prompt_tokens' in usage"
    assert 'total_tokens' in data['usage'], "Missing 'total_tokens' in usage"
    assert 'completion_tokens' in data['usage'], "Missing 'completion_tokens' in usage"

def get_model_token_limit(model):
    if model == "mistral-large-latest":
        return 128 * 1000
    elif model == "mistral-small-latest":
        return 32 * 1000
    elif model == "ministral-8b-latest":
        return 128 * 1000
    elif model == "ministral-3b-latest":
        return 128 * 1000
    else: 
        raise ValueError(f"Invalid model: {model}")

def is_valid_json(content):
    try:
        json.loads(content)
        return True
    except ValueError:
        return False

