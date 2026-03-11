from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import numpy as np
import zlib

def vectorize_abstract(csv_path):
    df = pd.read_csv(csv_path)
    combined_text = 'title: ' + df['title'] + " | Abstract: " + df['abstract']

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(combined_text.tolist(), show_progress_bar=True) #sentence transformer encode list
    vectors = np.array(embeddings).astype('float32')
    ids = df['hash_id'].astype('int64')        # convert the ids to int64

    return vectors, ids


dimension = 384
index = faiss.IndexFlatL2(dimension)        # brute force search by iterating through all the vectors
index_with_ids = faiss.IndexIDMap(index)    # faiss doesnt accept custom ids (rows is the id by default), so we need to wrap it with IndexIDMap

csv_path = 'D:/school/AI Project/RAG/dataset/arxiv_nlp.csv'  # you can custom this into your csv path
vectors, ids = vectorize_abstract(csv_path)
index_with_ids.add_with_ids(vectors, ids)

faiss.write_index(index_with_ids, "D:/school/AI Project/RAG/index/abstracts.index")
print("Success! Total paper in index: {index_with_ids.ntotal}")
