# Local Large Language Model (LLM) Inference Demo

This repository demonstrates how to run a **Large Language Model (LLM)** locally using the `ibm-granite/granite-3.1-2b-instruct` model from Hugging Face. The code will download the pre-trained model to your local machine and use it to generate text based on user input. This guide explains how to run the demo, install dependencies, and locate the downloaded model files.

## Table of Contents
1. [Overview](#overview)
2. [Installation Instructions](#installation-instructions)
3. [Usage](#usage)
4. [Model Download and Storage](#model-download-and-storage)
5. [Runtime Measurement](#runtime-measurement)
6. [Contributing](#contributing)
7. [License](#license)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This demo allows you to run a **Large Language Model (LLM)** locally on your machine using the `ibm-granite/granite-3.1-2b-instruct` model. The script performs the following actions:
- Downloads the model and tokenizer from the Hugging Face model hub.
- Accepts user input to generate text-based responses.
- Measures and outputs runtime for model inference.

This setup is ideal for testing LLMs locally without relying on cloud infrastructure.

---

## Installation Instructions

### Step 1: Set up Python Environment
Ensure you have Python 3.9 or later installed. You can download the latest version from [python.org](https://www.python.org/).

It is recommended to use a virtual environment to isolate dependencies:

```bash
python -m venv llm_inference_env
source llm_inference_env/bin/activate  # On Windows, use `llm_inference_env\Scripts\activate`
```

### Step 2: Install Required Libraries

#### 1. Install Hugging Face Transformers
```bash
pip install transformers
```

#### 2. Install PyTorch (GPU or CPU)
- **For CUDA-enabled GPUs:**  
  First, check your CUDA version:
  ```bash
  nvidia-smi
  ```
  Install the corresponding PyTorch version, e.g., for CUDA 12.1:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

- **For CPU-only setups:**
  ```bash
  pip install torch
  ```

To verify GPU compatibility:
```python
import torch
print(torch.cuda.is_available())  # Returns True if GPU is detected
```

---

## Usage

Once dependencies are installed, run the demo script:

```bash
python script_name.py
```

### Example
**Input:**  
*"Can you explain what a Large Language Model (LLM) is and how it works in simple terms?"*

**Output:**  
*"A Large Language Model (LLM) is a machine learning model designed to process and generate human language. It works by predicting the next word or phrase based on the input it receives, using patterns it has learned from vast amounts of text data."*

---

## Model Download and Storage

The script automatically downloads the `ibm-granite/granite-3.1-2b-instruct` model to your Hugging Face cache directory. Default locations:
- **Linux/macOS:** `~/.cache/huggingface/transformers`
- **Windows:** `C:\Users\your_username\.cache\huggingface\transformers`

You can change the cache directory by setting the `TRANSFORMERS_CACHE` environment variable:
```bash
export TRANSFORMERS_CACHE=/path/to/cache_directory  # Linux/macOS
set TRANSFORMERS_CACHE=C:\path\to\cache_directory  # Windows
```

---

## Runtime Measurement

The script measures the time taken for model inference. Example output:
```
Output: ['A Large Language Model (LLM) is...']
Runtime: 2.35 seconds
```

Performance depends on whether you use a CUDA-enabled GPU or CPU.

---

## Contributing

Contributions are welcome! Fork the repository, make your changes, and submit a pull request. Ensure your changes are well-documented and tested.

---

## License

This project is licensed under the **MIT License**. See the LICENSE file for more details.

---

## Troubleshooting

1. **Model download issues:** Ensure a stable internet connection and sufficient disk space.
2. **Slow performance:** Use a GPU for faster inference.
3. **Out of memory:** Reduce `max_new_tokens` or use CPU if GPU memory is insufficient.

Feel free to open an issue if you encounter further problems.

