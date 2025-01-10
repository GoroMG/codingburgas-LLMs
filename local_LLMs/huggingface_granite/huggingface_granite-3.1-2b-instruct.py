import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Record the start time to calculate the total runtime later
start_time = time.time()

# Set the device for running the model ("cuda" for GPU or "cpu" for CPU)
device = "cuda"  # Change to "cpu" if you don't have a GPU
# Specify the path to the pre-trained model
model_path = "ibm-granite/granite-3.1-2b-instruct"

# Load the tokenizer from the specified model path
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the model from the specified model path and assign it to the chosen device (GPU or CPU)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)

# Set the model to evaluation mode, which disables training-specific operations like dropout
model.eval()

# Create a conversation structure with system and user roles
chat = [
    {
        "role": "system",  # The system message sets the behavior of the assistant
        "content": "You are an assistant designed to provide concise, accurate, but brief answers. "
                   "Please follow the user's instructions carefully."
    },
    {
        "role": "user",  # The user message provides the input question
        "content": "Can you explain what a Large Language Model (LLM) is and how it works in simple terms?"
    }
]

# Apply the chat template to format the conversation for the model (don't tokenize yet and add a generation prompt)
chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# Tokenize the chat input to prepare it for the model
input_tokens = tokenizer(chat, return_tensors="pt").to(device)

# Generate the response from the model using the tokenized input
# max_new_tokens=200 ensures that the output will be up to 200 tokens long
output = model.generate(**input_tokens, max_new_tokens=200)

# Decode the output tokens into human-readable text
output = tokenizer.batch_decode(output)

# Print the model's response
print(output)

# Record the end time to calculate the runtime
end_time = time.time()

# Calculate the total runtime by subtracting the start time from the end time
runtime = end_time - start_time

# Print the total runtime in seconds
print(f"Runtime: {runtime} seconds")
