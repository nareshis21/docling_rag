from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
import os
import json
from typing import List, Dict

class LLMProcessor:
    def __init__(self):
        """Initialize embedding model and Groq LLM"""
        self.api_key = os.getenv("GROQ_API_KEY")

        # Use FastEmbed instead of SentenceTransformer
        self.embed_model = FastEmbedEmbeddings()

        self.llm = ChatGroq(
            model_name="mixtral-8x7b-32768",
            api_key=self.api_key
        )

    def format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into a structured context for the LLM"""
        context_parts = []
        for chunk in chunks:
            try:
                headings = json.loads(chunk['headings'])
                if headings:
                    context_parts.append(f"Section: {' > '.join(headings)}")
            except:
                pass

            if chunk['page']:
                context_parts.append(f"Page {chunk['page']}:")
            
            context_parts.append(chunk['text'])
            context_parts.append("-" * 40)

        return "\n".join(context_parts)

    def generate_answer(self, context: str, question: str) -> str:
        """Generate answer using structured context"""
        prompt = f"""Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know,if it is out of context say that it is out of context and also try to provide the answer and don't be rude.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

        return self.llm.invoke(prompt)
