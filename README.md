[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

[![Ollama](https://img.shields.io/badge/Ollama-black?style=flat&logo=ollama&logoColor=white)](https://ollama.com/)


# IPC Legal Chatbot

A web application for tracking packages and managing courier services.

## 📄 Overview

The IPC Legal Chatbot is a sophisticated Streamlit-based application designed to democratize access to legal information, specifically concerning the Indian Penal Code (IPC). It empowers users to quickly find simple and accurate explanations of IPC sections through an intuitive chat interface.

At its core, this chatbot utilizes a Retrieval-Augmented Generation (RAG) pipeline. This cutting-edge architecture combines the precision of information retrieval with the generative power of Large Language Models (LLMs). It leverages FAISS for efficient similarity search, Sentence-Transformers for converting text into meaningful embeddings, and Ollama to run the powerful Llama 3.1 model locally for generating concise and relevant legal explanations.

## ✨ Features

- Intelligent IPC Explanations: Get simple, easy-to-understand explanations for complex Indian Penal Code sections.

- Retrieval-Augmented Generation (RAG): Combines efficient document retrieval with advanced LLM generation for highly accurate and contextual answers.

- Local LLM Integration: Powered by the Llama 3.1 model, run locally via Ollama, ensuring data privacy and reducing reliance on cloud APIs.

- FAISS for Vector Search: Utilizes Facebook AI Similarity Search (FAISS) for lightning-fast retrieval of relevant IPC sections from your knowledge base.

- Sentence Embeddings: Employs Sentence-Transformers to create dense vector representations of legal texts and user queries.

- Interactive Streamlit UI: A clean and user-friendly web interface for a seamless conversational experience.

- Dynamic Data Handling: Automatically generates and caches FAISS index and metadata on the first run, optimizing subsequent performance.

- Comprehensive Logging: Detailed logging (to console and file) for monitoring and debugging, configured via logging_config.yaml.

## 🚀 Getting Started

You can run the IPC Legal Chatbot either directly on your machine or as a Docker container. Choose the method that best suits your environment.

Prerequisites

Ensure you have the following installed on your system:

- Git: For cloning the repository.

[Download & Install Git](https://git-scm.com/downloads)

- Python 3.8+: The application requires Python 3.8 or a newer version.

[Download Python](https://www.python.org/downloads/)

- Ollama: This is crucial for running the Llama 3.1 LLM locally.

[Install Ollama](https://ollama.com/download) - Follow the official instructions for your operating system.

- Docker: (Only if you choose Docker Deployment) For building and running the containerized application.

[Install Docker Engine](https://docs.docker.com/engine/install/) 

## 1. Clone the Repository

Begin by cloning the project repository to your local machine:

```
git clone https://github.com/Ashank007/ipc_legal_chatbot.git
cd ipc_legal_chatbot
```

## 2. Set Up Ollama and Llama 3.1 Model

- The chatbot relies on the Llama 3.1 model running via Ollama.

- Ensure Ollama is running: Make sure the Ollama server is active in your background. You can usually start it by simply installing it or running ollama serve if it's not set to auto-start.

- Download the Llama 3.1 model: Pull the specific model required by the application:

```
ollama run llama3.1
```
- This command will download the model if it's not already present and then start an interactive session. You can simply close this session (/bye) once the download is complete.

## 🏃‍♀️ Running the Application

You have two options to run the IPC Legal Chatbot:

### Option A: Run Directly (Local Development)

This method is ideal for development and testing without Docker.

Install Python Dependencies:

#### Create and activate a virtual environment (recommended)
```
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate   # On Windows
```

#### Install Python packages from requirements.txt
```
pip install -r requirements.txt
```
#### Run the Streamlit application:
```
streamlit run app.py
```

Your application should open in your default web browser at http://localhost:8501.

### Option B: Run with Docker (Recommended for Deployment)

Using Docker encapsulates all dependencies and ensures a consistent environment, making it ideal for deployment.

Build the Docker Image:
Navigate to the root directory of your project (where the Dockerfile is located) and run the following command:
```
docker build -t ipc_legal_chatbot .
```
This command will:

- Build the Docker image named ipc_legal_chatbot.

- Install all necessary system packages and Python dependencies.

- Copy your application code into the image.


Run the Docker Container:
Once the image is built, you can run the application using:

```
docker run -p 8501:8501 legal_chatbot
```

- docker run: Starts a new container from the legal_chatbot image.
- -p 8501:8501: Maps port 8501 of your local machine to port 8501 inside the container, where Streamlit is running.

After running the command, open your web browser and navigate to http://localhost:8501. You should see the IPC Legal Chatbot interface.

## 📂 Project Structure

```
ipc_legal_chatbot/
├── app.py                      # Main Streamlit application entry point
├── src/                        # Source code for modular components
│   ├── __init__.py             # Python package initializer
│   ├── data_processing.py      # Handles loading, parsing, and chunking of legal data
│   ├── rag_pipeline.py         # Implements the core Retrieval-Augmented Generation logic
│   ├── utils.py                # Contains various utility functions
│   └── config.py               # Stores configuration settings and constants
├── data/                       # Directory for legal data and generated artifacts
│   ├── legal_data.jsonl        # The raw legal knowledge base (JSON Lines format)
│   ├── ipc_faiss.index         # (Auto-generated) FAISS index for efficient retrieval
│   └── ipc_metadata.json       # (Auto-generated) Metadata associated with the FAISS index
├── requirements.txt            # Lists all Python dependencies
├── .env                        # Environment variables (e.g., API keys, model paths)
├── logging_config.yaml         # Configuration file for Python's logging module
├── Dockerfile                  # Docker instructions to build the container image
├── .dockerignore               # Specifies files/directories to exclude from Docker build context
└── README.md                   # This comprehensive project documentation
```

## 👨‍💻 Usage

- Open the App: Access the application in your web browser (typically at http://localhost:8501).

- Enter Your Query: Type a question or statement related to Indian Penal Code sections into the input field. Examples:

```
"What is IPC section 302?"

"Explain crimes against children under IPC."

"What are the punishments for theft?"
```

- Get Explanations: The chatbot will process your query, retrieve the most relevant IPC sections from its knowledge base, and generate a simplified explanation using the llama 3.1 model

## ⚙️ Dependencies

- Python 3.8+

- Streamlit: For the interactive web interface.

- Sentence-Transformers: For generating embeddings.

- FAISS-CPU: For efficient vector similarity search.

- PyTorch (CPU): Underlying framework for Sentence-Transformers.

- Ollama Python Library: To interact with the local Ollama server.

- Other utilities: langchain, pydantic, etc.

Refer to requirements.txt for the complete and exact list of Python dependencies.

## 📝 Notes

- Data Format: Ensure that legal_data.jsonl is correctly formatted with legal sections that the RAG pipeline can effectively process.

- Index Generation: The ipc_faiss.index and ipc_metadata.json files are automatically created and cached in the data/ directory upon the first run of the application. This significantly speeds up subsequent startups.

- Configuration: Adjust environment variables in the .env file for custom model paths, logging levels, or other application-specific settings.

- Logging: Application logs are configured via logging_config.yaml and are outputted to ipc_legal_chatbot.log as well as the console, aiding in debugging and monitoring.

## 🤝 Contributing

- Contributions, issues, and feature requests are highly welcome! Feel free to check the issues page.

## 📜 License

- This project is open-source and available under the MIT License.

📞 Contact

- For any questions or inquiries, please reach out to:

- [Email](ashankgupta.tech@gmail.com)

- [LinkedIn](https://www.linkedin.com/in/ashank-gupta-/)
