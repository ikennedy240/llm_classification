# LLM Text Classifier (ikennedy240/llm_classification)

This repository provides a simple, practical example of how you can use powerful Artificial Intelligence models—specifically Large Language Models (LLMs)—to automatically categorize and classify text data. Instead of manually labeling thousands of pieces of text, an LLM reads the text and assigns it to the correct category based on instructions we give it. Hopefully. Remember that you need to do a lot of checking and validation for this to be useful.

This project is designed to run locally on your computer, which means your private data never has to leave your machine.

---

## Prerequisites: Getting Your Local AI Server Running

The core script (`ollama_classify.py`) in this repository is set up to communicate with a local LLM server provided by **Ollama**.

**If you are new to this, setting up the server is the biggest hurdle! Here are your options:**

### Option 1: The Intended Way (Using Ollama)
This is the simplest path if you want to run the code exactly as written.

1.  **Install Ollama:** Follow the official installation guide for your operating system [here](https://ollama.com/download).
2.  **Pull a Model:** Once installed, open your terminal and pull a suitable model (e.g., `gemma4:e4b`):
    ```bash
    ollama pull gemma4:e4b
    ```

### Option 2: Advanced/Code Modification (Bypassing Servers)
If you want to avoid any external server setup and run classification purely through Python, you would need to modify `ollama_classify.py` to use a direct HuggingFace pipeline instead of the Ollama API call. This requires more advanced knowledge of Python libraries like `transformers`.

---

## Getting Started (Using Option 1)

### Step 1: Clone the Repository
```bash
git clone https://github.com/ikennedy240/llm_classification
cd llm_classification
```

### Step 2: Set Up Your Environment
You will need Python installed on your system. It is highly recommended to use a virtual environment manager like `uv`.

```bash
# Create and activate a virtual environment using uv or venv
uv venv
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate

# Install required dependencies (check the original repo for exact requirements.txt if available)
pip install -r requirements.txt # Assuming you have a requirements file
```

### Step 3: Modify the code
Make sure that the script points to the input data you want, and specify an output path.

### Step 4: Run the Classifier
Once your local LLM server is running and accessible, you can run the main script:

```bash
python ollama_classify.py 
```


