# 🤖 RAG Chatbot — Document Q&A Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload unstructured documents and ask questions in natural language, powered by **Groq's Llama 3.3 70B** for fast, context-aware response generation.

---

## 🚀 Demo

🔗 **Live Demo:** [https://rag-chatbot-kr26.onrender.com/](https://rag-chatbot-kr26.onrender.com/)

> Upload a document → Ask questions → Get intelligent, context-aware answers!

---

## 🧠 How It Works

User uploads document
↓
LlamaIndex chunks & embeds the text
↓
Embeddings stored in ChromaDB (vector database)
↓
User asks a question
↓
Semantic search retrieves relevant chunks
↓
Groq (Llama 3.3 70B) generates a context-aware answer
↓
Answer streamed to a custom chat UI

---


## ✨ Features

- 📄 Upload and query unstructured documents
- 💬 Ask natural language questions with context-aware answers
- 🔍 Semantic search and retrieval via ChromaDB embeddings
- ⚡ Fast, cloud-hosted inference using Groq (Llama 3.3 70B)
- 🖥️ Custom streaming chat UI for real-time responses
- 🐳 Containerized with Docker and deployed on Render

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq — Llama 3.3 70B (cloud-hosted) |
| RAG Framework | LlamaIndex |
| Vector Database | ChromaDB |
| Backend | FastAPI |
| UI | Custom streaming chat interface |
| Deployment | Docker, Render |
| Language | Python |

---

## 📦 Installation

```bash
# 1. Clone the repo
git clone https://github.com/aralsaran36-cell/Qwen-AI-Assistant.git
cd Qwen-AI-Assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
export GROQ_API_KEY=your_api_key_here

# 4. Run the backend
uvicorn app:app --reload
```

---

## 📁 Project Structure

RAG-Chatbot/
│
├── app.py # FastAPI backend
├── Dockerfile
├── README.md
└── requirements.txt
---


## 💡 Use Cases

- Chat with research papers
- Q&A on legal documents
- Study assistant for textbooks
- Internal knowledge base for companies

---

## 👨‍💻 Author

**Saravanan S** — Aspiring AI Engineer
- 📧 aralsaran36@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/feed/)
- 🐙 [GitHub](https://github.com/aralsaran36-cell)
