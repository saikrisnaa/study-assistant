import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from study_agent import StudyAgent
from document_processing import process_notes, setup_rag, get_answer, summarize_notes
from progress_visualization import visualize_progress

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

CHROMA_DIR = "./chroma_db"

def main():
    st.title("Personalized Study Assistant (Optimized for Large Files)")
    
    # Initialize StudyAgent with session state
    agent = StudyAgent()

    with st.sidebar:
        st.subheader("Upload Notes (Supports Large Files)")
        uploaded_file = st.file_uploader("Upload your notes (PDF or TXT)", type=["pdf", "txt"])
        raw_text = st.text_area("Or paste raw text notes here (optional):")
        force_reprocess = st.checkbox("Force reprocess (clear existing DB)", False)
        if st.button("Reset Chroma DB"):
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
                st.success("Chroma DB reset successfully!")

    vector_store = None
    qa_chain = None
    llm = None  # For fallback answers
    documents = None  # To store loaded documents for summarization

    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1]
        temp_file_path = f"temp_file.{file_extension}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        vector_store, documents = process_notes(temp_file_path, force_reprocess=force_reprocess)
        if vector_store:
            qa_chain, llm = setup_rag(vector_store)
            st.success("Notes processed successfully!")
    
    elif raw_text:
        vector_store, documents = process_notes(raw_text, file_type="raw_text", force_reprocess=force_reprocess)
        if vector_store:
            qa_chain, llm = setup_rag(vector_store)
            st.success("Raw text notes processed successfully!")

    if vector_store and qa_chain:
        # Question Input
        question = st.text_input("Ask a question about your notes:")
        if question:
            with st.spinner("Generating answer..."):
                answer = get_answer(qa_chain, llm, question)
                st.write("**Answer:**", answer)
        
        # Summarization Tool
        st.subheader("Summarize Your Notes")
        if st.button("Generate Summary"):
            if documents:
                with st.spinner("Generating summary..."):
                    summary_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
                    summary = summarize_notes(documents, summary_llm)
                    with st.expander("View Summary", expanded=True):
                        st.write(summary)
            else:
                st.error("No documents available for summarization.")

    # Study Plan and Progress Section
    st.subheader("Study Plan Suggestions")
    col1, col2 = st.columns(2)
    with col1:
        topic_progress = st.text_input("Topic for progress update")
        score = st.slider("Score (0-1)", 0.0, 1.0, 0.5)
        if st.button("Update Progress"):
            if topic_progress:
                agent.update_progress(topic_progress, score)
                st.success(f"Progress updated for {topic_progress}")
    with col2:
        suggestions = agent.suggest_study_plan()
        for suggestion in suggestions:
            st.write(suggestion)

    # Progress Visualization
    st.subheader("Progress Visualization")
    visualize_progress(agent)

if __name__ == "__main__":
    main()
