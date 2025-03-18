# Personalized Study Assistant

A Streamlit-based web app that helps users process and analyze study notes by providing features such as question-answering based on uploaded PDF or text documents, summarizing content, and tracking study progress. It leverages LangChain for document processing, OpenAI embeddings for search, and Plotly for visualizing study progress.

## Features

- **Upload Notes:** Supports PDF and TXT file uploads to process study materials.
- **OCR Preprocessing:** Removes unwanted OCR tags from text for clean processing.
- **Chunking Documents:** Large documents are split into smaller chunks for efficient querying.
- **Question Answering:** Use the uploaded documents to answer specific questions.
- **Summarization:** Generate concise summaries of the uploaded documents.
- **Study Progress Tracking:** Track progress by topic and visualize with Plotly charts.
- **Reprocessing & Reset:** Option to force reprocess documents and reset the vector database.

## Requirements

- Python 3.7+
- Streamlit
- LangChain
- OpenAI API Key
- Plotly
- Chroma (for vector store persistence)
- Python-dotenv
