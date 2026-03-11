import json
import pandas as pd
import zlib

FILE_DIRECTORY = 'D:/school/AI Project/RAG/dataset/'

def create_hash_id(id):
    return zlib.adler32(str(id).encode())

def collect_paper_in_category(file_path, category: str):
    with open(file_path, 'r') as f:
        for line in f:
            paper = json.loads(line)
            if category in paper['categories']:
                yield{
                    "paper_id": paper["id"],
                    "hash_id": create_hash_id(paper["id"]),
                    'title': paper['title'],
                    'authors': paper['authors'],    
                    'abstract': paper['abstract'],
                }

data_list = list(collect_paper_in_category(FILE_DIRECTORY + "arxiv-metadata-oai-snapshot.json", 'cs.CL'))
df = pd.DataFrame(data_list)
df.to_csv(FILE_DIRECTORY + 'arxiv_nlp.csv', index=False)
print('dataset successfully created')
