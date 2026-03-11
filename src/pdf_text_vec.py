import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
import json
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import OnlinePDFLoader, PyPDFLoader


def extract_pdf(paper_id):
    url = f"https://arxiv.org/pdf/{paper_id}"
    loader = PyPDFLoader(url)
    pages = loader.load()       # list of Document files. Document contains 2 main attributes, page_content and metadata (authors, page, title, ...)
    return pages

def chunk_paper(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(pages)
    return chunks

def vectorize_paper(chunked_text):
    embededing_model = SentenceTransformer("all-miniLM-L6-v2")

