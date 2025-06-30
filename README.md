# Thoughtful AI mini-RAG Chatbot

A Streamlit-based conversational RAG (Retrieval-Augmented Generation) chatbot focused on answering questions about Thoughtful AI using a hardcoded knowledge base.

## Features
- Uses OpenAI LLMs and ChromaDB for retrieval
- Loads system prompt from `system_prompt.md`
- Streams answers in real time
- Maintains per-session chat history

## Installation

1. **Clone the repository** and navigate to the project directory:
   ```sh
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. **Create and activate a virtual environment** (recommended):
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up your OpenAI API key:**
   - Create a `.env` file in the project root with the following line:
     ```
     OPENAI_API_KEY=your-openai-key-here
     ```

5. **Ensure your knowledge base and system prompt files are present:**
   - `data.json` (knowledge base)
   - `system_prompt.md` (system prompt)

## Running the App

Start the Streamlit app with:
```sh
streamlit run app.py
```

Open the provided local URL in your browser to interact with the chatbot.

## Running Tests

To run the integration tests:
```sh
python3 test_rag_service.py
```

---

**Note:**
- The chatbot only answers questions about Thoughtful AI using the provided knowledge base.
- For best results, ensure your OpenAI API key is valid and your `data.json` and `system_prompt.md` are well formatted.
