import os
import hashlib
import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document

CHROMA_DIR = "./chroma_db"

# Preprocess OCR-tagged text
def preprocess_ocr_text(text):
    lines = text.splitlines()
    cleaned = [line.strip() for line in lines if not (
        line.strip().startswith("<") and line.strip().endswith(">")
    ) and line.strip()]
    return "\n".join(cleaned)

# Compute file hash to detect changes
def get_file_hash(file_content):
    return hashlib.md5(file_content.encode('utf-8')).hexdigest()

# Process notes with fixed progress bar
def process_notes(file_path, file_type=None, force_reprocess=False):
    try:
        with st.spinner("Processing file..."):
            progress_bar = st.progress(0)
            
            # Load document
            if file_type == "raw_text":
                cleaned_text = preprocess_ocr_text(file_path)
                documents = [Document(page_content=cleaned_text)]
            elif file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
                documents = loader.load()
            else:
                raise ValueError("Unsupported file type. Use PDF or TXT.")
            
            progress_bar.progress(10)  # 10% after loading

            # Split into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            st.write(f"Split into {len(chunks)} chunks")

            # Check if reprocessing is needed
            file_hash = get_file_hash("".join(doc.page_content for doc in documents))
            hash_file = os.path.join(CHROMA_DIR, "file_hash.txt")
            if os.path.exists(hash_file):
                with open(hash_file, "r") as f:
                    stored_hash = f.read().strip()
            else:
                stored_hash = None

            embeddings = OpenAIEmbeddings()
            if os.path.exists(CHROMA_DIR) and stored_hash == file_hash and not force_reprocess:
                vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
                st.write("Loaded existing vector store")
            else:
                # Process in batches, fix progress calculation
                batch_size = 1000
                vector_store = None
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    if vector_store is None:
                        vector_store = Chroma.from_documents(batch, embeddings, persist_directory=CHROMA_DIR)
                    else:
                        vector_store.add_documents(batch)
                    progress_fraction = min(1.0, (i + len(batch)) / len(chunks))
                    progress_value = 10 + int(80 * progress_fraction)
                    progress_bar.progress(progress_value)
                
                # Save file hash
                if not os.path.exists(CHROMA_DIR):
                    os.makedirs(CHROMA_DIR)
                with open(hash_file, "w") as f:
                    f.write(file_hash)
                vector_store.persist()
                st.write("Created new vector store")

            progress_bar.progress(100)  # 100% when done
            return vector_store, documents  # Return documents for summarization
    except Exception as e:
        st.error(f"Error processing notes: {str(e)}")
        return None, None

# Setup RAG pipeline
def setup_rag(vector_store):
    try:
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5})
        )
        return qa_chain, llm  # Return llm for fallback use
    except Exception as e:
        st.error(f"Error setting up RAG: {str(e)}")
        return None, None

# Get answer with relevance check and LLM fallback
def get_answer(qa_chain, llm, question):
    try:
        # Retrieve documents with similarity scores
        retrieved_docs_with_scores = qa_chain.retriever.vectorstore.similarity_search_with_score(question, k=5)
        retrieved_docs = [doc for doc, score in retrieved_docs_with_scores]
        scores = [score for doc, score in retrieved_docs_with_scores]
        
        # Debugging: Show retrieved chunks and their similarity scores
        st.write("Retrieved chunks and scores:", [(doc.page_content[:100] + "...", score) for doc, score in retrieved_docs_with_scores])
        
        # Check relevance using a similarity score threshold (lower score = more similar)
        relevant_docs = [doc for doc, score in retrieved_docs_with_scores if score < 0.4]  # Lower score = more similar
        
        if not relevant_docs:
            # Fallback to LLM if no relevant documents are found
            st.write("Topic not found in document. Generating answer with LLM...")
            prompt = f"Provide a general answer to the following question: {question}"
            return llm.predict(prompt)
        else:
            # Use document-specific answer if relevant content is found
            response = qa_chain.run(question)
            return response
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Sorry, I couldn’t process your question."

# Summarize notes
def summarize_notes(documents, llm):
    try:
        # Combine all document content into a single string, truncated if too long
        full_content = "\n".join([doc.page_content for doc in documents])
        max_tokens = 4000  # Rough limit for GPT-4o-mini context; adjust as needed
        if len(full_content) > max_tokens:
            full_content = full_content[:max_tokens]  # Truncate to fit context window
        
        # Directly use the LLM with a custom prompt
        prompt = f"Summarize the following content concisely, focusing on key points:\n\n{full_content}"
        summary = llm.predict(prompt)  # Use predict for direct LLM call
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return "Sorry, I couldn’t generate a summary."
