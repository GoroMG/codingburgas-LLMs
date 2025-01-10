
# AI Assistant with Ollama

This project is a simple web application that interacts with the Ollama and a model via Streamlit. The application allows real-time question and answer functionality utilizing Ollama's `generate` endpoint.

## Requirements

To run this project, you'll need the following:

- Python 3.9 or later.
- `streamlit` and `requests` libraries.
- Ollama must be running locally on the default port `11434` to handle the `generate` endpoint. Check http://localhost:11434/ and if you don't see `Ollama is running`. Start it in a terminal:

    ```
    ollama serve
    ```


## Start the Streamlit application:

  ```
  streamlit run app.py
  ```


## How It Works

The core function of the app is `generate_response`, which communicates with Ollama's `generate` endpoint to get the model's response. Here’s a breakdown of how it works:

- The function uses `requests.post` to make a POST request to Ollama's `generate` endpoint (`/api/generate`).
- The request payload includes:
  - `model`: The model version to use (e.g., `llama3.2`).
  - `prompt`: The user’s input or question.
  - `context`: The ongoing context of the conversation, which allows the model to keep track of prior exchanges.
  
- Ollama’s `generate` endpoint returns a stream of JSON blobs. These blobs are iterated through using `response.iter_lines()`:
  - The response body is checked for the `response` field, which contains the model’s answer.
  - Once the full response is collected, it is printed to the screen.
