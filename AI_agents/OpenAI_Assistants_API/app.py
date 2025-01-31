import openai
import time
from dotenv import load_dotenv
import os
import gradio as gr
import logging

# Set up logging to capture detailed information about the process
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (e.g., API keys)
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = openai.Client()


# Function to handle math questions and get responses from the assistant
def solve_math_problem(problem):
    """
    Solves the given math problem by interacting with the OpenAI Assistant.

    Args:
        problem (str): The math problem provided by the user, e.g., '3x + 11 = 14'.

    Returns:
        str: The assistant's response, which includes the solution or an error message.
    """
    try:
        # Log the received problem
        logger.info(f"Received problem: {problem}")

        # Step 1: Create an Assistant
        assistant = client.beta.assistants.create(
            name="Math Solver",
            instructions="You are a personal math tutor. Write and run code to answer math questions. "
                         "Always reply in the same language as the question. "
                         "If the question is in Bulgarian, respond in Bulgarian!",
        tools=[{"type": "code_interpreter"}],
            model="gpt-4-1106-preview"
        )
        logger.info(f"Assistant created with ID: {assistant.id}")

        # Step 2: Create a Thread for communication
        thread = client.beta.threads.create()
        logger.info(f"Thread created with ID: {thread.id}")

        # Step 3: Add the user message to the thread
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"I need to solve the equation `{problem}`. Can you help me? "
                    f"Please respond in the same language as the question."
        )
        logger.info("User message sent to assistant.")

        # Step 4: Run the Assistant to solve the problem
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions="Please provide the exact steps to solve the equation"
        )
        logger.info(f"Run started with ID: {run.id}")

        # Step 5: Wait for the assistant's response to be ready
        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run_status.status == "completed":
                logger.info("Assistant completed the task.")
                break
            elif run_status.status == "failed":
                error_message = f"Run failed: {run_status.last_error}"
                logger.error(error_message)
                return error_message
            time.sleep(2)  # wait for 2 seconds before checking again

        # Step 6: Retrieve and return the assistant's response
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        logger.info(f"Retrieved {len(messages.data)} messages from the thread.")

        # Loop through messages and find the assistant's response
        full_response = ""
        for message in messages.data:  # Process messages sequentially, not reversed
            if message.role == "assistant":
                # Log the structure of the message content for debugging
                logger.info(f"Message content: {message.content}")

                # Check if the content is a list and contains LaTeX or text
                if isinstance(message.content, list):
                    for block in message.content:
                        # Ensure that we have a 'text' attribute, and access its 'value'
                        if hasattr(block, 'text') and hasattr(block.text, 'value'):
                            # Extract text value from LaTeX or plain text
                            text_value = block.text.value
                            full_response += text_value
                        else:
                            logger.warning(f"Block does not contain valid text: {block}")
                else:
                    logger.warning(f"Unexpected message content format: {type(message.content)}")

                logger.info("Assistant's response found.")
                break

        # Return the full response or an error if no response was found
        if not full_response:
            logger.warning("No valid response found.")
            return "No valid response received from the assistant."
        return full_response

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return f"Error: {e}"


# Create a Gradio interface for user input and output
def create_ui():
    """
    Creates the user interface using Gradio, where users can enter a math problem,
    and the assistant will return the solution.
    """
    with gr.Blocks() as demo:
        # Display a title and instructions on how to use the assistant
        gr.Markdown("# Math Solver Assistant\nSolve mathematical problems with a personal math tutor.")
        gr.Markdown("### Instructions: Enter a math problem, and the assistant will solve it for you.")

        # Input box for math problem and output box for the assistant's response
        with gr.Row():
            problem_input = gr.Textbox(label="Enter your math problem", placeholder="e.g., 3x + 11 = 14")
            result_output = gr.Textbox(label="Solution", interactive=False)

        # Button that triggers the assistant to solve the math problem
        solve_button = gr.Button("Solve Equation")

        # Button interaction to process the problem and display the result
        solve_button.click(solve_math_problem, inputs=[problem_input], outputs=[result_output])

    # Launch the Gradio interface
    logger.info("Launching the Gradio UI.")
    demo.launch()


# Run the UI
if __name__ == "__main__":
    logger.info("Starting Math Solver Assistant.")
    create_ui()