from locust import HttpUser, task, between, events
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")
BASE_URL = os.getenv("BASE_URL")
API_ENDPOINT = "/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
MODEL = 'mistral-large-latest'

# Default values for test parameters
DEFAULT_SPAWN_RATE = 2
DEFAULT_RUN_TIME = "1m"
DEFAULT_HOST = BASE_URL

@events.init_command_line_parser.add_listener
def add_custom_arguments(parser):
    """Set default values for Locust's command-line arguments."""
    parser.set_defaults(
        spawn_rate=DEFAULT_SPAWN_RATE,
        run_time=DEFAULT_RUN_TIME,
        host=DEFAULT_HOST,
    )

class MistralUser(HttpUser):
    """Simulates a user sending requests to the Mistral API."""
    wait_time = between(3, 6) 

    @task
    def send_request(self):
        """Task to send a chat completion request."""
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello, Locust is testing you!"}],
        }
        with self.client.post(
            API_ENDPOINT,
            json=payload,
            headers=HEADERS,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code {response.status_code}")
