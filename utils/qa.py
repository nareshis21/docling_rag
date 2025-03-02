import logging
from ingestion import DocumentProcessor
from llm import LLMProcessor


class QAEngine:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.llm_processor = LLMProcessor()

    def query(self, question: str, k: int = 5) -> str:
        """Query the document using semantic search and generate an answer"""
        query_embedding = self.llm_processor.embed_model.encode(question)

        # Corrected ChromaDB query syntax
        results = self.processor.index.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        # Extracting results properly
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


# def main():
#     logging.basicConfig(level=logging.INFO)

#     processor = DocumentProcessor()
    
#     pdf_path = "sample/InternLM.pdf"
#     processor.process_document(pdf_path)

#     qa_engine = QAEngine()
#     question = "What are the main features of InternLM-XComposer-2.5?"
#     answer = qa_engine.query(question)

#     print("\nAnswer:")
#     print("=" * 80)
#     print(answer)


# if __name__ == "__main__":
#     main()
