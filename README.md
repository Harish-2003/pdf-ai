# 📄 Chat with your PDF (RAG)

A small Retrieval-Augmented Generation (RAG) app. Upload a PDF, ask questions about it, and get answers based only on the document. Built with LangChain, Chroma, local sentence-transformers embeddings, a Groq-hosted LLM, and a Gradio UI.

## How it works

1. The PDF is loaded and split into chunks (800 characters, 100 overlap).
2. Each chunk is embedded locally with `all-MiniLM-L6-v2` and stored in Chroma.
3. For each question, the 3 most similar chunks are retrieved.
4. The chunks and the question are sent to the LLM, which answers using only that context.

## Requirements

- Python 3.10 or newer
- A free [Groq API key](https://console.groq.com)

## Setup

**1. Clone the repo**

```
git clone https://github.com/Harish-2003/pdf-ai.git
cd pdf-ai
```

**2. Create and activate a virtual environment**

```
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux, use `source venv/bin/activate` instead.

**3. Install the dependencies**

```
pip install -r requirements.txt
```

The first install is large because it includes PyTorch (via sentence-transformers), so it may take a few minutes.

**4. Add your API key**

Create a file named `.env` in the project folder with this line:

```
GROQ_API_KEY=your_key_here
```

Do not use quotes, and never commit this file (it is already listed in `.gitignore`).

## Run the app

```
python pdf_chat.py
```

Then open the local address printed in the terminal:

```
http://127.0.0.1:7860
```

## Using the app in Gradio

1. Upload a PDF in the **Upload a PDF** box.
2. Wait for the message **✅ PDF indexed! Ask me anything about it.** The first upload is slower because the embedding model loads.
3. Type a question in the chat box and press Enter.
4. To use a different PDF, upload it and wait for the indexed message again.

## Notes

- Use `demo.launch()` in `pdf_chat.py` for local use. `share=True` creates a public link that lets anyone use your Groq key.
- Scanned PDFs (images of text) are not supported, because the loader reads only real text.
- The first run downloads the embedding model (about 90 MB) from Hugging Face. After that it works from the local cache.
- The HF_TOKEN warning in the terminal is harmless.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ImportError: ... sentence_transformers` | Activate the venv, then run `pip install -r requirements.txt` |
| `GROQ_API_KEY` not found or 401 error | Check that `.env` exists in the project folder and the key has no quotes or spaces |
| Chat says "Please upload a PDF first" | Wait for the "PDF indexed" message, or re-upload after restarting the app |
| `model not found` error | Groq retires models over time. Pick a current one from the Groq models page and update `model=` in `pdf_chat.py` |
