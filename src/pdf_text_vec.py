import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pdf_text_cleaning import clean_paper

# ---------------------------------------------------------
# 1. WORKER FUNCTION (Runs in parallel)
# ---------------------------------------------------------
def worker_extract_and_chunk(paper_id, hash_id):
    """
    Handles the slow network downloads and text parsing.
    Returns plain text strings to avoid Multiprocessing pickle errors.
    """
    try:
        url = f"https://arxiv.org/pdf/{paper_id}"
        loader = PyPDFLoader(url)
        pages = loader.load()
        
        if not pages:
            return paper_id, hash_id, None, "No pages found"

        for page in pages:
            page.page_content = clean_paper(page.page_content)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        
        # Extract purely the text strings to pass safely back to the main process
        chunk_texts = [chunk.page_content for chunk in chunks]
        
        return paper_id, hash_id, chunk_texts, None

    except ValueError as e:
        if "404" in str(e):
            return paper_id, hash_id, None, "Error 404: Paper not found on arXiv."
        return paper_id, hash_id, None, f"Value Error: {e}"
    except Exception as e:
        return paper_id, hash_id, None, f"Unexpected error: {e}"


# ---------------------------------------------------------
# 2. MAIN PROCESS (Runs sequentially, handles Model & FAISS)
# ---------------------------------------------------------
if __name__ == '__main__':
    CSV_PATH = 'dataset/arxiv_nlp.csv'
    INDEX_DIRECTORY = 'index'
    INDEX_SAVE_PATH = Path(INDEX_DIRECTORY) / "pdf_nlp_chunks.index"
    JSONL_SAVE_PATH = Path(INDEX_DIRECTORY) / "nlp_chunks_metadata.jsonl"
    
    os.makedirs(INDEX_DIRECTORY, exist_ok=True)

    print("Loading SentenceTransformer model...")
    model = SentenceTransformer("all-miniLM-L6-v2") # Ensure GPU usage
    dimensions = model.get_embedding_dimension()

    index = faiss.IndexFlatL2(dimensions)
    index_with_ids = faiss.IndexIDMap(index)
    
    df = pd.read_csv(CSV_PATH)
    
    processed_papers = set()
    global_chunk_id = 0

    # ---------------------------------------------------------
    # CHECKPOINT RECOVERY: Recover from existing JSONL
    # ---------------------------------------------------------
    if INDEX_SAVE_PATH.exists() and JSONL_SAVE_PATH.exists():
        print("Found existing FAISS index and JSONL file. Loading index directly...")
        index_with_ids = faiss.read_index(str(INDEX_SAVE_PATH))
        loaded_ntotal = index_with_ids.ntotal
        
        batch_text = []
        batch_ids = []
        RECOVERY_BATCH_SIZE = 10000

        with open(JSONL_SAVE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): 
                    continue
                
                data = json.loads(line)
                processed_papers.add(str(data["paper_id"]))
                
                # Update our global counter to make sure new chunks don't overwrite old ones
                global_chunk_id = max(global_chunk_id, data["chunk_id"] + 1)
                
                if data["chunk_id"] < loaded_ntotal:
                    continue
                
                # Accumulate a small batch in memory for missing items
                batch_text.append(data["text"])
                batch_ids.append(data["chunk_id"])

                # Once the batch is full, encode it and clear RAM
                if len(batch_text) >= RECOVERY_BATCH_SIZE:
                    print(f"-> Indexing missing batch of {len(batch_text)} chunks (Current total in FAISS: {index_with_ids.ntotal})...")
                    embeddings = model.encode(batch_text, show_progress_bar=False, batch_size=256)
                    embeddings = np.array(embeddings).astype('float32')
                    index_with_ids.add_with_ids(embeddings, np.array(batch_ids, dtype=np.int64))
                    
                    # CRUCIAL: Empty the lists immediately to free up RAM
                    batch_text.clear()
                    batch_ids.clear()

        # Process any remaining chunks left over at the end of the file
        if batch_text:
            print(f"-> Indexing final missing batch of {len(batch_text)} chunks...")
            embeddings = model.encode(batch_text, show_progress_bar=True, batch_size=256)
            embeddings = np.array(embeddings).astype('float32')
            index_with_ids.add_with_ids(embeddings, np.array(batch_ids, dtype=np.int64))
            batch_text.clear()
            batch_ids.clear()
            
        print(f"✅ FAISS index safely restored to {index_with_ids.ntotal} vectors without crashing RAM.")

    elif JSONL_SAVE_PATH.exists():
        print("Found existing JSONL file. Rebuilding FAISS index via streaming...")
        
        batch_text = []
        batch_ids = []
        RECOVERY_BATCH_SIZE = 10000  # Process 10k chunks at a time to save RAM

        with open(JSONL_SAVE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): 
                    continue
                
                data = json.loads(line)
                processed_papers.add(str(data["paper_id"]))
                
                # Update our global counter to make sure new chunks don't overwrite old ones
                global_chunk_id = max(global_chunk_id, data["chunk_id"] + 1)
                
                # Accumulate a small batch in memory
                batch_text.append(data["text"])
                batch_ids.append(data["chunk_id"])

                # Once the batch is full, encode it and clear RAM
                if len(batch_text) >= RECOVERY_BATCH_SIZE:
                    print(f"-> Indexing batch of {len(batch_text)} chunks (Current total in FAISS: {index_with_ids.ntotal})...")
                    embeddings = model.encode(batch_text, show_progress_bar=False, batch_size=256)
                    embeddings = np.array(embeddings).astype('float32')
                    index_with_ids.add_with_ids(embeddings, np.array(batch_ids, dtype=np.int64))
                    
                    # CRUCIAL: Empty the lists immediately to free up RAM
                    batch_text.clear()
                    batch_ids.clear()

        # Process any remaining chunks left over at the end of the file
        if batch_text:
            print(f"-> Indexing final recovery batch of {len(batch_text)} chunks...")
            embeddings = model.encode(batch_text, show_progress_bar=True, batch_size=256)
            embeddings = np.array(embeddings).astype('float32')
            index_with_ids.add_with_ids(embeddings, np.array(batch_ids, dtype=np.int64))
            batch_text.clear()
            batch_ids.clear()
            
        print(f"✅ FAISS index safely restored to {index_with_ids.ntotal} vectors without crashing RAM.")

    # Filter out already processed papers
    remaining_df = df[~df['paper_id'].astype(str).isin(processed_papers)]
    print(f"Total papers in dataset: {len(df)} | Already processed: {len(processed_papers)} | Remaining: {len(remaining_df)}")

    # ---------------------------------------------------------
    # MEMORY-SAFE THROTTLED PIPELINE
    # ---------------------------------------------------------
    print("Starting processing pool...")
    success_count = 0
    
    # We slice the remaining dataframe into small batches (e.g., 500 papers)
    # This prevents the futures queue from overloading your RAM.
    BUFFER_SIZE = 500 
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for chunk_idx in range(0, len(remaining_df), BUFFER_SIZE):
            batch_df = remaining_df.iloc[chunk_idx:chunk_idx + BUFFER_SIZE]
            
            # Submit only a small batch to the executor
            futures = {
                executor.submit(worker_extract_and_chunk, str(row['paper_id']), int(row['hash_id'])): str(row['paper_id'])
                for _, row in batch_df.iterrows()
            }
            
            # Resolve this batch immediately
            for future in as_completed(futures):
                try:
                    paper_id, hash_id, chunk_texts, error = future.result()
                    
                    if error or not chunk_texts:
                        continue

                    chunk_texts = [str(text) for text in chunk_texts if text and isinstance(text, str) and str(text).strip()]
                    
                    if not chunk_texts:
                        continue

                    embeddings = model.encode(chunk_texts, show_progress_bar=True)
                    embeddings = np.array(embeddings).astype('float32')

                    num_chunks = len(chunk_texts)
                    chunk_ids = np.arange(global_chunk_id, global_chunk_id + num_chunks, dtype=np.int64)

                    index_with_ids.add_with_ids(embeddings, chunk_ids)

                    with open(JSONL_SAVE_PATH, 'a', encoding='utf-8') as f:
                        for i, text in enumerate(chunk_texts):
                            metadata = {
                                "chunk_id": int(chunk_ids[i]),
                                "hash_id": hash_id,
                                "paper_id": paper_id,
                                "text": text
                            }
                            f.write(json.dumps(metadata) + "\n")
                    
                    global_chunk_id += num_chunks
                    success_count += 1

                except Exception as e:
                    print(f"⚠️ Error processing a paper: {e}")
            
            # INTERMITTENT SAVE: Save FAISS to disk after every batch of 500 papers
            faiss.write_index(index_with_ids, str(INDEX_SAVE_PATH))
            print(f"💾 Checkpoint Saved! Total papers in this session: {success_count} | Total vectors in FAISS: {index_with_ids.ntotal}")

    # --- Finalize ---
    faiss.write_index(index_with_ids, str(INDEX_SAVE_PATH))
    print("\n🎉 Run complete! FAISS Index fully updated and synchronized.")