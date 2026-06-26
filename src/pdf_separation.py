import json
import pandas as pd
import zlib
from pathlib import Path

DATASET_DIRECTORY = 'dataset'   # change to '.' if saved in current directory
ARXIV_CODE = 'cs.CL'
OUTPUT_CSV_PATH = 'arxiv_nlp.csv'

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

data_list = list(collect_paper_in_category(DATASET_DIRECTORY / "arxiv-metadata-oai-snapshot.json", ARXIV_CODE))
df = pd.DataFrame(data_list)
df.to_csv(Path(DATASET_DIRECTORY) / OUTPUT_CSV_PATH, index=False)
print('dataset successfully created')

# output: DATASET_DIRECTORY/OUTPUT_CSV_PATH