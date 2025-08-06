# Offline Chatbot using Ollama and NLTK

[![Python](https://img.shields.io/badge/python-3.6+-blue.svg)]() [![NLTK](https://img.shields.io/badge/NLTK-3.8-green.svg)]() [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A fully offline, Python-based chatbot combining classical NLP (NLTK + TF-IDF) with a local LLM (Mistral-7B via Ollama) to deliver fast, context-aware conversations without internet dependency.

---

## 🚀 Features

* **TF-IDF Fallback**: Instantly resolves common queries by matching user inputs against a base sentence corpus using scikit-learn’s TF-IDF and cosine similarity.
* **Offline LLM Integration**: Leverages Mistral‑7B through Ollama, running locally, for complex or novel queries when TF-IDF fails.
* **Adaptive Typing Simulation**: Simulates human typing at a configurable delay (`0.01s` by default), adjustable at runtime via the `set speed` command.
* **Greeting Detection**: Recognizes and responds to greetings (e.g., “hello,” “hi,” “sup”) with randomized friendly replies.
* **Contextual Memory**: Maintains the last **10** messages (user + assistant) to preserve dialogue coherence in LLM responses.

---

## 🏗 Architecture

1. **Preprocessing (NLTK)**

   * **Tokenization & Lemmatization**: Cleans and normalizes text to base forms, improving match accuracy.
   * **Punctuation Removal**: Strips punctuation to avoid mismatches.

2. **TF-IDF Matching**

   * Builds a TF-IDF matrix over predefined sentences + user query.
   * Computes cosine similarity; if **score ≥ 0.1**, returns the top-matching sentence.

3. **LLM Fallback (Mistral‑7B via Ollama)**

   * Formats the recent conversation history as a prompt:

     ```text
     User: ...
     Assistant: ...
     User: <latest input>
     Assistant:
     ```
   * Sends the prompt to Ollama's local server and retrieves the generated response.
   * Uses `temperature=0.7` and `top_p=0.9` for balanced creativity and relevance.

4. **Typing Simulation**

   * Prints output character-by-character with a delay loop (`time.sleep(typing_delay)`).

5. **Command Handling**

   * **`set speed`** — Adjust typing delay.
   * **`bye`**, **`thanks`** — Graceful exit.

---

## 🧾 Prerequisites

* **Python 3.6+**
* **Ollama** (local installation & models): [Download Guide](https://ollama.ai/download)
* **NLTK** and required corpora (`punkt`, `wordnet`, `averaged_perceptron_tagger`)
* **scikit-learn**

All Python dependencies are listed in `requirements.txt`.

---

## 🖥️ System Requirements

* **Operating System**: Windows 10/11, macOS (11+), or Linux (Ubuntu 18.04+ recommended)
* **CPU**: 4+ cores (recommended for running LLM locally)
* **RAM**: Minimum 8 GB (16 GB+ recommended for smoother LLM operation)
* **Disk Space**: At least 5 GB free (for models and NLTK corpora)
* **Python Version**: 3.6 or higher
* **Network**: No internet required for chatbot operation after initial setup (Ollama and models run locally)

---

## ⚙ Installation

1. **Clone the Repo**

   ```bash
   git clone https://github.com/Boominathan2355/Offline-Chatbot-using-Ollama-and-NLTK.git
   cd Offline-Chatbot-using-Ollama-and-NLTK
   ```

2. **Create and Activate Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\\Scripts\\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data**

   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
   ```

5. **Ensure Ollama Is Running**

   ```bash
   # Example: start Ollama daemon/service
   ollama start
   ```

---

## ▶️ Usage

```bash
python chatbot.py
```

* **Greet**: Say “hello” or “hi” to get an instant greeting.
* **Ask Questions**: General queries trigger TF-IDF; complex ones invoke the LLM.
* **`set speed`**: Type to adjust typing delay.
* **`bye`/`thanks`**: End the session.

---

## 🗂 Project Structure

```text
Offline-Chatbot-using-Ollama-and-NLTK/
├── chatbot.py           # Main entrypoint
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
├── README.md            # This document
└── nltk_data/           # NLTK corpora downloaded at runtime
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Commit changes: `git commit -m "Add YourFeature"`
4. Push branch: `git push origin feature/YourFeature`
5. Open a pull request

---

## 📄 License

MIT License © Boominathan Alagirisamy. See [LICENSE](LICENSE).

