---

# **Docling RAG QA App**  

This is a **Retrieval-Augmented Generation (RAG) Question-Answering (QA) system** built using **Docling** for document processing, **ChromaDB** for vector storage, and **Streamlit** for a user-friendly web UI.  

The app extracts and indexes text from multiple document formats (**PDF, DOCX, PPTX, HTML, TXT**), allowing users to **upload documents**, **ask questions**, and receive **LLM-generated answers** with **relevant document content.**  

---

## **Features**  
✅ **Upload & process multiple documents** (PDF, DOCX, PPTX, HTML, TXT)  
✅ **Automated text extraction**  
✅ **Fast & scalable document search** using **ChromaDB**  
✅ **Semantic search with embeddings** (FastEmbed)  
✅ **LLM-powered answers** with **context-based retrieval**  
✅ **Streamlit Web UI** for easy interaction  

---

## **Installation**  
### **1️⃣ Clone the repository**  
```bash
git clone https://github.com/nareshis21/docling_rag.git
cd docling_rag
```

### **2️⃣ Create a virtual environment & install dependencies**  
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## **Usage**  
### **1️⃣ Run the Streamlit UI**  
```bash
streamlit run app.py
```
This will launch the **web interface** in your browser.  

### **2️⃣ Upload Documents**  
- Use the **"Upload Files"** section to add PDFs, DOCX, PPTX, HTML, or TXT files.  
- The system will extract text, chunk documents, and store embeddings in **ChromaDB**.  

### **3️⃣ Ask Questions**  
- Enter a question in the text box.  
- The system will **retrieve relevant document chunks** and generate an **LLM-powered answer**.  

---

## **Example Query**  
📝 **User:**  
*"What are the key insights from the report?"*  

🤖 **Answer:**  
*"The report highlights the following key insights... (AI-generated answer)"*  

---

## **Configuration**  
Modify `config.py` to adjust parameters like:  
- **ChromaDB storage path**  
- **Embedding model**  
- **Max chunk size**  

---

## **Future Enhancements**  
🚀 **Add table OCR support** (Next release)  
🚀 **Implement source tracking (document name & page number)** (Next release)  
🚀 **Improve retrieval ranking with hybrid search**  
🚀 **Extend support for more document types**  
🚀 **Add chat memory for multi-turn interactions**  

---

## **Contributors**  
👤 **Your Name** – Developer  

---

## **License**  
📜 MIT License  

---

This **Docling RAG QA App** makes document-based Q&A **fast, efficient, and user-friendly** with **Streamlit UI**! 🚀 Let me know if you need modifications!
