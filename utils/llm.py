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
        prompt = f"""Based on the following excerpts from a document:

{context}

Please answer this question: {question}

Make use of the section information and page numbers in your answer when relevant.
"""
        return self.llm.invoke(prompt)
