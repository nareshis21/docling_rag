import streamlit as st
import os
import json
import base64
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from utils.ingestion import DocumentProcessor
from utils.llm import LLMProcessor
from utils.qa import QAEngine

# Configure Streamlit page
st.set_page_config(page_title="AI-Powered Document QA", layout="wide")

# # Background Image
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(
#         f"""
#         <style>
#             .stApp {{
#                 background-image: url(data:image/png;base64,{encoded_string.decode()});
#                 background-size: cover;
#             }}
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

# # Path to background image
# image_bg = "./image/background.jpeg"  # Change this path accordingly
# add_bg_from_local(image_bg)

# Initialize document processing & AI components
document_processor = DocumentProcessor()
llm_processor = LLMProcessor()
qa_engine = QAEngine()

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

# Sidebar for file upload
st.sidebar.header("Upload a PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

# Initialize chat memory
memory_storage = StreamlitChatMessageHistory(key="chat_messages")
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", human_prefix="User", chat_memory=memory_storage, k=5
)

# Document upload & processing
if uploaded_file and "document_uploaded" not in st.session_state:
    pdf_path = os.path.join("temp", uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.sidebar.success("File uploaded successfully!")

    with st.spinner("Processing document..."):
        document_processor.process_document(pdf_path)

    st.sidebar.success("Document processed successfully!")
    st.session_state["document_uploaded"] = True

# Chat interface layout
st.markdown("<h2 style='text-align: center;'>AI Chat Assistant</h2>", unsafe_allow_html=True)
st.markdown("---")

# Display chat history
for message in memory_storage.messages:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# User input at the bottom
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Store user message in memory
    memory_storage.add_user_message(user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Generating response..."):
        if st.session_state.get("document_uploaded", False):
            answer = qa_engine.query(user_input)
        else:
            answer = llm_processor.generate_answer("", user_input)
            st.warning("No document uploaded. This response is generated from general AI knowledge and may not be document-specific.")

    # Store AI response in memory
    memory_storage.add_ai_message(answer)

    with st.chat_message("assistant"):
        st.markdown(answer)
