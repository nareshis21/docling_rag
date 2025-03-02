import logging
from utils.ingestion import DocumentProcessor
from utils.llm import LLMProcessor


class QAEngine:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.llm_processor = LLMProcessor()
        self.collection = self.processor.client.get_or_create_collection("document_chunks")  # Fix

    def query(self, question: str, k: int = 5) -> str:
        """Query the document using semantic search and generate an answer"""

        # ✅ Correct method for getting embeddings
        query_embedding = self.llm_processor.embed_model.embed_query(question)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        chunks = []
        for i in range(len(results["documents"][0])):  # Iterate over top-k results
            chunks.append({
                "text": results["documents"][0][i],
                "headings": results["metadatas"][0][i].get("headings", "[]"),
                "page": results["metadatas"][0][i].get("page"),
                "content_type": results["metadatas"][0][i].get("content_type")
            })

        print(f"\nRelevant chunks for query: '{question}'")
        print("=" * 80)

        context = self.llm_processor.format_context(chunks)
        print(context)

        return self.llm_processor.generate_answer(context, question)
