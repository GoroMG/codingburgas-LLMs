import json
import requests
import streamlit as st

# Define the model to be used
MODEL_NAME = 'llama3.2'  # Update this to your desired model version

# Function to generate responses from the model
def generate_response(prompt):
    """
    Send a prompt to the model's API and get a response.

    Args:
        prompt (str): User input to generate a response for.

    Returns:
        str: The response from the model.
    """
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': MODEL_NAME,
                'prompt': prompt,
            },
            stream=True
        )
        response.raise_for_status()

        result = ""
        for line in response.iter_lines():
            data = json.loads(line)
            result += data.get('response', '')

            if 'error' in data:
                st.error(f"Error: {data['error']}")
                break

            if data.get('done', False):
                break

        return result

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""


# Streamlit app definition
def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("AI Assistant with Ollama")
    st.write("Interact with local model in real-time.")

    # User input text box
    user_input = st.text_input("Enter your prompt:", placeholder="Type something...")

    if st.button("Generate Response") and user_input:
        with st.spinner("Generating response..."):
            response = generate_response(user_input)
        if response:
            st.text_area("Model Response:", value=response, height=200)


if __name__ == "__main__":
    main()
