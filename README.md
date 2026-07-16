🤖 Qwen AI Assistant — RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) chatbot that lets you upload documents and ask questions in natural language. Powered by Qwen 2.5 (local LLM via Ollama), LlamaIndex, and ChromaDB for fast semantic search.

🚀 Demo

Upload a PDF → Ask questions → Get intelligent answers from your document!
🧠 How It Works

User uploads document
        ↓
LlamaIndex chunks & embeds the text
        ↓
Embeddings stored in ChromaDB (vector DB)
        ↓
User asks a question
        ↓
Semantic search finds relevant chunks
        ↓
Qwen 2.5 (via Ollama) generates the answer
        ↓
Answer displayed in Streamlit UI
✨ Features

📄 Upload PDF / text documents
💬 Ask natural language questions on your document
🔍 Semantic search using ChromaDB embeddings
🧩 Local LLM (Qwen 2.5) via Ollama — 100% offline & private
⚡ Fast retrieval with LlamaIndex
🖥️ Clean Streamlit UI with session-based chat history
🛠️ Tech Stack

Component	Technology
LLM	Qwen 2.5 (via Ollama)
RAG Framework	LlamaIndex
Vector Database	ChromaDB
Embeddings	Semantic Embeddings
UI	Streamlit
Language	Python
📦 Installation

# 1. Clone the repo
git clone https://github.com/aralsaran36-cell/Qwen-AI-Assistant.git
cd Qwen-AI-Assistant

# 2. Install dependencies
pip install llama-index chromadb streamlit ollama

# 3. Pull Qwen model via Ollama
ollama pull qwen2.5

# 4. Run the app
streamlit run app.py
📁 Project Structure

Qwen-AI-Assistant/
│
├── Gwen AI Assistant.ipynb   # Main notebook
├── README.md
└── requirements.txt
💡 Use Cases

Chat with research papers
Q&A on legal documents
Study assistant for textbooks
Internal knowledge base for companies
👨‍💻 Author

Saravanan S — Aspiring AI Engineer

📧 aralsaran36@gmail.com
💼 LinkedIn
🐙 GitHub
