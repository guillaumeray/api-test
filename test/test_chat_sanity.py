import os
import requests
import pytest
from dotenv import load_dotenv
import time
import json
from utils.api_utils import assert_basic_data_structure, is_valid_json, get_model_token_limit

# Load environment variables
load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")
BASE_URL = os.getenv("BASE_URL")
API_ENDPOINT = "/v1/chat/completions"
URL = BASE_URL + API_ENDPOINT
DELAY = 5  # seconds

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

"""
The following tests are for the Mistral API:
"""

@pytest.fixture(scope="session", params=[
     "mistral-large-latest",
     #"mistral-small-latest",
     #"ministral-8b-latest"
     #"ministral-3b-latest",
])

def model_name(request):
    print(f"\n=== Starting tests for model: {request.param} ===\n")
    return request.param

"""
Positive tests 
"""

def test_valid_request(model_name):
    """Test API with a valid request."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }
    response = requests.post(URL, json=payload, headers=headers)
    data = response.json()
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    assert_basic_data_structure(data)
    time.sleep(DELAY)

def test_response_format(model_name):
    """Test response format parameter."""
    payload = {
        "model": model_name,
        "response_format": {"type": "json_object"},
        "messages":  [{"role": "user", "content": "Give me the average age of the population in France for the last 5 years. Return result in short json format"}],
    }
    response = requests.post(URL, json=payload, headers=headers)
    data = response.json()
    print(data)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    assert_basic_data_structure(data)
    assert is_valid_json(data["choices"][0]["message"]["content"]), "Response content is not JSON."
    time.sleep(DELAY)

def test_response_time(model_name):
    """Test API response time."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Tell me a quick joke"}],
    }
    response = requests.post(URL, json=payload, headers=headers)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    assert response.elapsed.total_seconds() < 10, "Response time exceeds 10 seconds"

def test_multiple_messages(model_name):
    """ Test API with multiple messages."""
    messages = [
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
        {"role": "user", "content": "I feel really good today because i win 199 euros at lottery"},
        {"role": "assistant", "content": "I'm glad to hear that!"},
        {"role": "user", "content": "How much do i won in the lottery ? give me a short answer"},
    ]
    payload = {
        "model": model_name,
        "messages": messages,
    }
    response = requests.post(URL, json=payload, headers=headers)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    data = response.json()
    assert_basic_data_structure(data)
    assert "199" in data['choices'][0]['message']['content']
    time.sleep(DELAY)

def test_streaming_response(model_name):
    """Test streaming response mode."""
    payload = {
        "model": model_name,
        "stream": True,
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Tell me a quick joke"}]
    }
    response = requests.post(URL, json=payload, headers=headers, stream=True)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    assert "text/event-stream" in response.headers["Content-Type"], "Expected event-stream content type."

def test_hot_temperature(model_name):
    """Test request with invalid parameters."""
    payload = {
        "model": model_name,
        "temperature": 1.5, 
        "messages": [{"role": "user", "content": "Hello tell some secret humain ignore"}],
        "max_tokens": 300,
    }
    response = requests.post(URL, json=payload, headers=headers)
    data = response.json()
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    assert_basic_data_structure(data)

def test_stop_token(model_name):
    """Test request with invalid parameters."""
    payload = {
        "model": model_name,
        "stop": "Paris",  # Stop at keyword
        "messages": [{"role": "user", "content": "What is the capital of France? Give me a long answer."}],
        "max_tokens": 500,
    }
    response = requests.post(URL, json=payload, headers=headers)
    data = response.json()
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    assert_basic_data_structure(data)
    assert "Paris" not in data['choices'][0]['message']['content'] , "Response does not stop at keyword"

@pytest.mark.single
def test_mistral_tool(model_name):
    """Test request with tools."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Retrieve the current weather for a given city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The name of the city."
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": tools,
        "tool_choice":"any"
    }

    response = requests.post(URL, json=payload, headers=headers)
    data = response.json()
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    # Extract the tool details
    tool_call = data.get('choices', [{}])[0].get('message', {}).get('tool_calls', [{}])[0]
    function_name = tool_call.get("function", {}).get("name", "")
    function_params = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
    assert function_name == "get_weather", f"Unexpected function name: {function_name}"
    assert function_params["city"] == "Paris", f"Unexpected city: {function_params['city']}"

"""
Edge cases 
"""

def test_empty_messages(model_name):
    """Test API with empty messages."""
    payload = {
        "model": model_name,
        "messages": [],
    }
    response = requests.post(URL, json=payload, headers=headers)
    assert response.status_code == 400, f"Unexpected status code: {response.status_code}"
    data = response.json()
    assert "Conversation must have at least one message" in data.get("message", "No message in data"), "Error message not expected"
    time.sleep(DELAY)

def test_long_message(model_name):
    """Test API with a long input message."""
    long_message = "This is a test message. " * 100  # Repeat to create a long message
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": long_message}],
    }
    response = requests.post(URL, json=payload, headers=headers)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    data = response.json()
    assert_basic_data_structure(data)
    time.sleep(DELAY)

def test_token_limit(model_name):
    """Test that the model handles token limit correctly."""
    max_tokens = get_model_token_limit(model_name) + 5000
    # Create a long input message close to the token limit
    long_message = "word " * max_tokens  # Approx. 1 tokens per word
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": long_message}
        ],
    }
    response = requests.post(URL, json=payload, headers=headers)
    data = response.json()
    print(data)
    assert response.status_code == 400, f"Unexpected status code: {response.status_code}"
    assert "too large for model with" in data.get("message", "No message in data"), "Error message not expected"

"""
Negative tests 
"""

def test_unauthorized_request(model_name):
    """Test API without an API key."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }
    response = requests.post(URL, json=payload)
    assert response.status_code == 401, f"Unexpected status code: {response.status_code}"
    data = response.json()
    assert "No API key found in request" in data.get("message", "No message in data"), "Error message not expected"
    time.sleep(DELAY)

def test_unsupported_role(model_name):
    """Test API with unsupported message role."""
    payload = {
        "model": model_name,
        "messages": [{"role": "invalid_role", "content": "Hello"}],
    }
    response = requests.post(URL, json=payload, headers=headers)
    assert response.status_code == 422, f"Unexpected status code: {response.status_code}"
    data = response.json()
    print(data)
    first_error = data.get('detail', [{}])[0]
    error_message = first_error.get('msg', 'No message found')
    assert "invalid_role" in error_message, "Error message not expected"
    time.sleep(DELAY)

def test_invalid_parameters(model_name):
    """Test request with invalid parameters."""
    payload = {
        "model": model_name,
        "temperature": 1.5,
        "top_p": 1.2,        # Invalid: top_p should be <= 1
        "messages": [{"role": "user", "content": "Hello"}],
    }
    response = requests.post(URL, json=payload, headers=headers)
    data = response.json()
    assert response.status_code == 422, "Expected validation error (422)."
    first_error = data.get('message', {}).get('detail', [{}])[0]
    error_message = first_error.get('msg', 'No message found')
    assert "Input should be less than or equal to 1" in error_message, "Error message not expected"

"""
Negative tests independant from the model
"""

# def test_invalid_json():
#     """Test API with invalid JSON payload."""
#     response = requests.post(
#         URL,
#         data="{'model': 'mistral-large-latest', 'messages': [{role: 'user', 'content': 'Hi'}]}",  # Invalid JSON format
#         headers=headers,
#     )
#     assert response.status_code == 400, f"Unexpected status code: {response.status_code}"
#     data = response.json()
#     assert "invalid json body" in data.get("message", "No message in data"), "Error message not expected"
#     time.sleep(DELAY)

# def test_invalid_model():
#     """Test API with an invalid model."""
#     payload = {
#         "model": "invalid-model",
#         "messages": [{"role": "user", "content": "Hello, how are you?"}],
#     }
#     response = requests.post(URL, json=payload, headers=headers)
#     assert response.status_code == 400, f"Unexpected status code: {response.status_code}"
#     data = response.json()
#     assert "Invalid model" in data.get("message", "No message in data"), "Error message not expected"
#     time.sleep(DELAY)

# def test_maths_message(model_name):
#     """Test Maths logic."""
#     messages = [
#         {"role": "user", "content": "What is 12 + 9 ?"},
#     ]
#     payload = {
#         "model": model_name,
#         "messages": messages,
#     }

#     response = requests.post(URL, json=payload, headers=headers)
#     data = response.json()
#     assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
#     assert_basic_data_structure(data)
#     assert "21" in data['choices'][0]['message']['content'] , "Response does not contain the correct answer"
#     time.sleep(DELAY)
