import streamlit as st
import os
import json
from utils.ingestion import DocumentProcessor
from utils.llm import LLMProcessor
from utils.qa import QAEngine

# Set up Streamlit page
st.set_page_config(page_title="AI-Powered Document QA", layout="wide")
st.title("📄 AI-Powered Document QA")

# Initialize processors
document_processor = DocumentProcessor()
llm_processor = LLMProcessor()
qa_engine = QAEngine()

# File uploader
st.sidebar.header("Upload a PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    # Save file to a temporary path
    pdf_path = f"temp/{uploaded_file.name}"
    os.makedirs("temp", exist_ok=True)
    
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.sidebar.success("✅ File uploaded successfully!")

    # Process the document
    with st.spinner("🔄 Processing document..."):
        document_processor.process_document(pdf_path)
    
    st.sidebar.success("✅ Document processed successfully!")

    # Query input
    question = st.text_input("Ask a question from the document:", placeholder="What are the key insights?")

    if st.button("🔍 Search & Answer"):
        if question:
            with st.spinner("🧠 Searching for relevant context..."):
                answer = qa_engine.query(question)
            
            st.subheader("📝 Answer:")
            st.write(answer)

        else:
            st.warning("⚠️ Please enter a question.")

# Footer
st.markdown("---")
st.caption("🤖 Powered by ChromaDB + Groq LLM | Built with ❤️ using Streamlit")
