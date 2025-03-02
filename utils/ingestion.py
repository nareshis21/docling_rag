import json
import time
import os
from pathlib import Path
from typing import Dict, Any, List
from tempfile import mkdtemp

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import chromadb


class DocumentProcessor:
    def __init__(self):
        """Initialize document processor with necessary components"""
        self.setup_document_converter()
        self.embed_model = FastEmbedEmbeddings()
        self.client = chromadb.PersistentClient(path=mkdtemp())  # Persistent storage

    def setup_document_converter(self):
        """Configure document converter with advanced processing capabilities"""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options.lang = ["en"]
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.MPS
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )

    def extract_chunk_metadata(self, chunk) -> Dict[str, Any]:
        """Extract essential metadata from a chunk"""
        metadata = {
            "text": chunk.text,
            "headings": [],
            "page_info": None,
            "content_type": None
        }

        if hasattr(chunk, 'meta'):
            if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                metadata["headings"] = chunk.meta.headings

            if hasattr(chunk.meta, 'doc_items'):
                for item in chunk.meta.doc_items:
                    if hasattr(item, 'label'):
                        metadata["content_type"] = str(item.label)

                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, 'page_no'):
                                metadata["page_info"] = prov.page_no

        return metadata

    def process_document(self, pdf_path: str) -> Any:
        """Process document and create searchable index with metadata"""
        print(f"Processing document: {pdf_path}")
        start_time = time.time()

        result = self.converter.convert(pdf_path)
        doc = result.document

        chunker = HybridChunker(tokenizer="jinaai/jina-embeddings-v3")
        chunks = list(chunker.chunk(doc))

        processed_chunks = []
        for chunk in chunks:
            metadata = self.extract_chunk_metadata(chunk)
            processed_chunks.append(metadata)

        print("\nCreating vector database...")
        collection = self.client.get_or_create_collection(name="document_chunks")

        documents = []
        embeddings = []
        metadata_list = []
        ids = []

        for idx, chunk in enumerate(processed_chunks):
            embedding = self.embed_model.encode(chunk['text'])
            documents.append(chunk['text'])
            embeddings.append(embedding)
            metadata_list.append({
                "headings": json.dumps(chunk['headings']),
                "page": chunk['page_info'],
                "content_type": chunk['content_type']
            })
            ids.append(str(idx))

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata_list
        )

        processing_time = time.time() - start_time
        print(f"\nDocument processing completed in {processing_time:.2f} seconds")
        return collection
