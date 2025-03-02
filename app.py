import streamlit as st
import os
import base64
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from utils.ingestion import DocumentProcessor
from utils.llm import LLMProcessor
from utils.qa import QAEngine

# Configure Streamlit page
st.set_page_config(page_title="AI-Powered Document QA", layout="wide")

# Function to encode image in Base64 for avatars
def encode_image(image_path):
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode()

# Load avatar images
user_avatar = encode_image("./icons/user.jpg")  # Change path if needed
ai_avatar = encode_image("./icons/ai.jpg")

# Initialize document processing & AI components
document_processor = DocumentProcessor()
llm_processor = LLMProcessor()
qa_engine = QAEngine()

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

# Sidebar for file upload
st.sidebar.header("📂 Upload a Document")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "docx", "html", "pptx", "txt"])

# Initialize chat memory
memory_storage = StreamlitChatMessageHistory(key="chat_messages")
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", human_prefix="User", chat_memory=memory_storage, k=5
)

# Document upload & processing
if uploaded_file and "document_uploaded" not in st.session_state:
    file_path = os.path.join("temp", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.sidebar.success("✅ File uploaded successfully!")

    with st.spinner("🔄 Processing document..."):
        document_processor.process_document(file_path)

    st.sidebar.success("✅ Document processed successfully!")
    st.session_state["document_uploaded"] = True

# Chat UI Header
st.title("💬SOP QA")
st.divider()

# Display chat history
for message in memory_storage.messages:
    role = "user" if message.type == "human" else "assistant"
    avatar = user_avatar if role == "user" else ai_avatar  # Assign appropriate avatar

    with st.chat_message(role, avatar=f"data:image/jpeg;base64,{avatar}"):
        if role == "assistant":
            # Display AI response as a properly formatted Markdown block (copiable)
            st.markdown(f"```\n{message.content}\n```")  
        else:
            st.markdown(message.content)

# User input at the bottom
user_input = st.chat_input("Type your message here...")

if user_input:
    memory_storage.add_user_message(user_input)

    # Display user message
    with st.chat_message("user", avatar=f"data:image/jpeg;base64,{user_avatar}"):
        st.markdown(user_input)

    with st.spinner("🤖 Thinking..."):
        if st.session_state.get("document_uploaded", False):
            answer = qa_engine.query(user_input)
        else:
            answer = llm_processor.generate_answer("", user_input)
            st.warning("⚠️ No document uploaded. Response is from general AI knowledge.")

    memory_storage.add_ai_message(answer)

    # Display AI response as a Markdown block (copiable)
    with st.chat_message("assistant", avatar=f"data:image/jpeg;base64,{ai_avatar}"):
        st.markdown(f"```\n{answer.content}\n```")  
