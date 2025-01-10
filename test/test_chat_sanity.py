import os
import requests
import pytest
from dotenv import load_dotenv
import time
import json
from utils.api_utils import assert_basic_data_structure, is_valid_json, get_model_token_limit
from typing import Generator

"""
This module contains a suite of tests for the Mistral API, focusing on chat completion endpoints.
The tests cover various scenarios including positive tests, edge cases, and negative tests to ensure
the API behaves as expected under different conditions.

Fixtures:
    model_name: Provides different model names for the tests.
    after_each_test: Introduces a delay after each test to avoid hitting rate limits.
Tests:
    - test_valid_request: Tests API with a valid request.
    - test_response_format: Tests the response format parameter for a given model.
    - test_response_time: Tests the API response time for a given model.
    - test_multiple_messages: Tests the API with multiple messages to ensure it responds correctly.
    - test_streaming_response: Tests streaming response mode.
    - test_hot_temperature: Tests request with high temperature parameters.
    - test_stop_token: Tests the API request with invalid parameters to ensure the response stops at the specified keyword.
    - test_mistral_tool: Tests the Mistral tool functionality by sending a request to the API and verifying the response.
    - test_empty_messages: Tests the API's response when provided with an empty list of messages.
    - test_long_message: Tests the API with a long input message.
    - test_token_limit: Tests that the model handles token limit correctly.
    - test_unauthorized_request: Tests sending a request to the API without an API key and verify the response.
    - test_unsupported_role: Tests API with unsupported message role.
    - test_invalid_parameters: Tests request with invalid parameters.
    - test_invalid_json: Tests API with invalid JSON payload.
    - test_invalid_model: Tests API with an invalid model.
"""

# Load environment variables
load_dotenv()

# Retrieve API key and base URL from environment variables
API_KEY = os.getenv("MISTRAL_API_KEY")
BASE_URL = os.getenv("BASE_URL")

# Define the API endpoint and complete URL
API_ENDPOINT = "/v1/chat/completions"
URL = BASE_URL + API_ENDPOINT

# Set delay between tests to avoid hitting rate limits
DELAY = 3  # seconds

# Define headers for the API requests
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

"""
Hook fixture
"""

@pytest.fixture(scope="session", params=[
    "mistral-large-latest",
    # "mistral-small-latest",
    # "ministral-8b-latest",
    # "ministral-3b-latest",
])
def model_name(request: pytest.FixtureRequest) -> Generator[str, None, None]:
    """
    Fixture to provide different model names for the tests.
    This fixture runs once per session and iterates over the provided model names.
    
    Args:
       request (pytest.FixtureRequest): The pytest request object.

    Yields:
       str: The current model name for the test.
    """
    print(f"\n=== Starting tests for model: {request.param} ===\n")
    yield request.param

@pytest.fixture(autouse=True)
def after_each_test():
    """
    Fixture to run after each test. Introduces a delay to avoid hitting rate limits.
    """
    yield
    time.sleep(DELAY)

"""

Positive cases

"""

def test_valid_request(model_name: str) -> None:
    """
    Test API with a valid request.
    
    This test sends a valid request to the API and checks if the response status code is 200.
    It also verifies that the response data has the basic expected structure.
    
    Args:
        model_name (str): The name of the model to be used in the request.
    """
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }
    response = requests.post(URL, json=payload, headers=headers)
    data = response.json()
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    assert_basic_data_structure(data)

def test_response_format(model_name: str) -> None:
    """
    Test the response format parameter for a given model.

    Args:
        model_name (str): The name of the model to test.

    Raises:
        AssertionError: If the response status code is not 200.
        AssertionError: If the response content is not a valid JSON object.
    """
    payload = {
        "model": model_name,
        "response_format": {"type": "json_object"},
        "messages":  [{"role": "user", "content": "Give me the average age of the population in France for the last 5 years. Return result in short json format"}],
    }
    response = requests.post(URL, json=payload, headers=headers)
    data = response.json()
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    assert_basic_data_structure(data)
    assert is_valid_json(data["choices"][0]["message"]["content"]), "Response content is not JSON."

def test_response_time(model_name: str) -> None:
    """
    Test the API response time for a given model.

    Args:
        model_name (str): The name of the model to test.

    Raises:
        AssertionError: If the response status code is not 200 or if the response time exceeds 10 seconds.
    """
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Tell me a quick joke"}],
    }
    response = requests.post(URL, json=payload, headers=headers)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    assert response.elapsed.total_seconds() < 10, "Response time exceeds 10 seconds"

def test_multiple_messages(model_name: str) -> None:
    """
    Test the API with multiple messages to ensure it responds correctly.

    Args:
        model_name (str): The name of the model to be tested.

    Raises:
        AssertionError: If the response status code is not 200 or if the expected content is not found in the response.

    This test sends a series of messages to the API and checks if the response contains the expected content.
    The messages simulate a conversation between a user and an assistant. The test verifies that the assistant's
    response includes the number "199", which is mentioned in the user's message about winning the lottery.
    """
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

def test_streaming_response(model_name: str) -> None:
    """
    Test streaming response mode.

    This function sends a POST request to the specified URL with a payload that 
    includes the model name, streaming flag, maximum tokens, and a message. 
    It asserts that the response status code is 200 and that the content type 
    of the response is 'text/event-stream'.

    Args:
        model_name (str): The name of the model to be tested.

    Raises:
        AssertionError: If the response status code is not 200 or if the content 
                        type is not 'text/event-stream'.
    """
    payload = {
        "model": model_name,
        "stream": True,
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Tell me a quick joke"}]
    }
    response = requests.post(URL, json=payload, headers=headers, stream=True)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    assert "text/event-stream" in response.headers["Content-Type"], "Expected event-stream content type."

def test_hot_temperature(model_name: str) -> None:
    """
    Test request with invalid parameters.

    This test sends a POST request to the API with a payload with max temperature value (1.5).
    It verifies that the response status code is 200 and checks the basic structure of the response data.

    Args:
        model_name (str): The name of the model to be tested.

    Raises:
        AssertionError: If the response status code is not 200 or if the response data structure is invalid.
    """
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

def test_stop_token(model_name: str) -> None:
    """
    Test the API request with invalid parameters to ensure the response stops at the specified keyword.

    Args:
        model_name (str): The name of the model to be used in the request.

    Raises:
        AssertionError: If the response status code is not 200 or if the response does not stop at the specified keyword.

    This test sends a request to the API with a payload containing a stop keyword and checks if the response
    stops at the specified keyword. It verifies the status code of the response and ensures the response
    content does not include the stop keyword.
    """
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

def test_mistral_tool(model_name: str) -> None:
    """
    Test the Mistral tool functionality by sending a request to the API and verifying the response.

    Args:
        model_name (str): The name of the model to be tested.

    Raises:
        AssertionError: If the response status code is not 200.
        AssertionError: If the function name in the response is "get_weather".
        AssertionError: If the city parameter in the function arguments is "Paris".
    """
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

def test_empty_messages(model_name: str) -> None:
    """
    Test the API's response when provided with an empty list of messages.

    Args:
        model_name (str): The name of the model to be tested.

    Asserts:
        The response status code is 400.
        The error message indicates that the conversation must have at least one message.
    """
    payload = {
        "model": model_name,
        "messages": [],
    }
    response = requests.post(URL, json=payload, headers=headers)
    assert response.status_code == 400, f"Unexpected status code: {response.status_code}"
    data = response.json()
    assert "Conversation must have at least one message" in data.get("message", "No message in data"), "Error message not expected"

def test_long_message(model_name: str) -> None:
    """
    Test the API with a long input message.

    This function sends a POST request to the API with a long message to test
    how the API handles large inputs. It verifies that the response status code
    is 200 and checks the basic structure of the response data.

    Args:
        model_name (str): The name of the model to be tested.

    Raises:
        AssertionError: If the response status code is not 200 or if the response
                        data does not have the expected basic structure.
    """
    long_message = "This is a test message. " * 100  # Repeat to create a long message
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": long_message}],
    }
    response = requests.post(URL, json=payload, headers=headers)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    data = response.json()
    assert_basic_data_structure(data)

def test_token_limit(model_name: str) -> None:
    """
    Test that the model handles token limit correctly.

    This test sends a request to the model with an input message that exceeds the model's token limit.
    It verifies that the model returns a 400 status code and an appropriate error message indicating
    that the input is too large for the model.

    Args:
        model_name (str): The name of the model to be tested.

    Raises:
        AssertionError: If the response status code is not 400 or the error message is not as expected.
    """
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
    assert response.status_code == 400, f"Unexpected status code: {response.status_code}"
    assert "too large for model with" in data.get("message", "No message in data"), "Error message not expected"

"""

Negative cases 

"""

def test_unauthorized_request(model_name: str) -> None:
    """
    Test sending a request to the API without an API key and verify the response.

    Args:
        model_name (str): The name of the model to be used in the request payload.

    Raises:
        AssertionError: If the response status code is not 401 or if the error message
                        does not contain "No API key found in request".

    This test ensures that the API correctly handles unauthorized requests by checking
    that the response status code is 401 and that the error message indicates the absence
    of an API key.
    """
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }
    response = requests.post(URL, json=payload)
    assert response.status_code == 401, f"Unexpected status code: {response.status_code}"
    data = response.json()
    assert "No API key found in request" in data.get("message", "No message in data"), "Error message not expected"

def test_unsupported_role(model_name: str) -> None:
    """
    Test API with unsupported message role.

    This test sends a POST request to the API with a payload containing an unsupported
    message role and verifies that the API responds with a 422 status code and an 
    appropriate error message.

    Args:
        model_name (str): The name of the model to be used in the payload.

    Raises:
        AssertionError: If the response status code is not 422 or if the error message 
                        does not contain "invalid_role".
    """
    payload = {
        "model": model_name,
        "messages": [{"role": "invalid_role", "content": "Hello"}],
    }
    response = requests.post(URL, json=payload, headers=headers)
    assert response.status_code == 422, f"Unexpected status code: {response.status_code}"
    data = response.json()
    first_error = data.get('detail', [{}])[0]
    error_message = first_error.get('msg', 'No message found')
    assert "invalid_role" in error_message, "Error message not expected"

def test_invalid_parameters(model_name: str) -> None:
    """
    Test request with invalid parameters.

    This test sends a POST request with a payload containing invalid parameters
    to the specified URL and checks if the response status code is 422, indicating
    a validation error. It also verifies that the error message in the response
    indicates that the 'top_p' parameter should be less than or equal to 1.

    Args:
        model_name (str): The name of the model to be included in the payload.

    Raises:
        AssertionError: If the response status code is not 422 or if the error
                        message does not indicate the expected validation error.
    """
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

Negative cases independant from the model

"""

def test_invalid_json() -> None:
    """Test API with invalid JSON payload."""
    response = requests.post(
        URL,
        data="{'model': 'mistral-large-latest', 'messages': [{role: 'user', 'content': 'Hi'}]}",  # Invalid JSON format
        headers=headers,
    )
    assert response.status_code == 400, f"Unexpected status code: {response.status_code}"
    data = response.json()
    assert "invalid json body" in data.get("message", "No message in data"), "Error message not expected"

def test_invalid_model() -> None:
    """Test API with an invalid model."""
    payload = {
        "model": "invalid-model",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }
    response = requests.post(URL, json=payload, headers=headers)
    assert response.status_code == 400, f"Unexpected status code: {response.status_code}"
    data = response.json()
    assert "Invalid model" in data.get("message", "No message in data"), "Error message not expected"
