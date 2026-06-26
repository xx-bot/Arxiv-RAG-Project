import sqlite3
import json

def create_db(json_path, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            text TEXT
        )
    ''')

    batch_count = 0
    with open(json_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            text = data.get('text', '')
            cursor.execute('INSERT INTO chunks (id, text) VALUES (?, ?)', (i, text))

            if i%50000 == 0:
                conn.commit()
                print(f"batch {batch_count} complete!")
                batch_count+=1
    
    conn.commit()
    conn.close()
    print("Database built successfully!!")

create_db('index/nlp_chunks_metadata.jsonl', 'dataset/db_chunks.sqlite')