# Mistral API Test

# Mistral API Test Suite

This repository contains a suite of tests for the Mistral API, focusing on chat completion endpoints. The tests cover various scenarios including positive tests, edge cases, and negative tests to ensure the API behaves as expected under different conditions.

## Prerequisites

- Python 3.12 or higher
- `pip` (Python package installer)

## Installation

1. Clone the repository:

    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a virtual environment:

    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:

    - On Windows:

        ```sh
        venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```sh
        source venv/bin/activate
        ```

4. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

## Environment Variables

Create a `.env` file in the root directory of the project and add the following environment variables:

```
MISTRAL_API_TOKEN=your_api_token_here
```

## Running Api Tests

To run the test located in `test/test_chat_sanity.py`, use the following command:

```sh
pytes test/test_chat_sanity.py 
```

If needed its possible to have html report:

```sh
pytest --html=report/api-report.html test/test_chat_sanity.py
```

## Running Performance Tests

To run the performance test located in `test/locustfile.py`, use the following command:

```sh
locust --user 5 -f test/locustfile.py --headless --html report/report.html
```

This command runs the performance tests using Locust with 5 simulated users in headless mode and generates an HTML report named `report.html`.

## Directory Structure

```
├── README.md
├── requirements.txt
├── .env
└── test
    ├── test_chat_sanity.py
    ├── locustfile.py
    └── utils
```