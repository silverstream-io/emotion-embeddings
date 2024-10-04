import csv, dotenv, itertools, os, time
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from pinecone import Pinecone
from tqdm import tqdm
import argparse
import json

dotenv.load_dotenv()
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# Add these variables at the top of the file, after the imports
TOKEN_LIMIT = 3000
SLEEP_TIME = 60  # seconds
token_count = 0

def chunks(iterable, batch_size=200):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def start_batch(file_id: str) -> str:
    """Start a batch embedding job."""
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/embeddings",
        completion_window="24h"
    )
    return batch_response.id

def check_batch(batch_id: str) -> str:
    """Check the status of a batch embedding job."""
    response = client.batches.retrieve(batch_id)
    return response.status, response.errors

def fetch_embeddings(batch_id: str) -> Tuple[List[Dict[str, Any]], str]:
    """Fetch embeddings from a completed batch job."""
    response = client.batches.retrieve(batch_id)
    if response.status != "finished":
        raise ValueError("Batch job not finished")
    output_file = response.result_files[0]
    content = client.files.content(output_file)
    embeddings = [json.loads(line) for line in content.splitlines()]
    return embeddings, response.input_file_id

def do_upsert(embeddings: List[Dict[str, Any]], metadata_file: str):
    """Upsert a batch of vectors to Pinecone."""
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    vectors = []
    for emb in embeddings:
        id = emb['custom_id'].split('-')[-1]
        vectors.append((emb['custom_id'], emb['embedding'], metadata[id]))
    
    for ids_vectors_chunk in chunks(vectors, batch_size=200):
        index.upsert(vectors=ids_vectors_chunk, namespace='emotions')

def process_csv(file_path: str, jsonl_output: str, metadata_output: str, max_entries: int = 50000) -> int:
    """Process a CSV file and create a JSONL file for batch processing and a JSON file for metadata."""
    entry_count = 0
    metadata = {}
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        ids = []
        reader = csv.DictReader(csvfile)
        emotion_columns = reader.fieldnames[9:37]  # Columns from "admiration" to "neutral"

        with open(jsonl_output, 'w', encoding='utf-8') as jsonlfile:
            for row in tqdm(reader, desc=f"Processing {file_path}"):
                if not row['id'] in ids:
                    ids.append(row['id'])
                else:
                    continue
                if entry_count >= max_entries:
                    break
                id = row['id']
                text = row['text']
                custom_id = f"request-{os.path.basename(file_path)[:-4]}-{id}"
                jsonl_entry = json.dumps({"custom_id": custom_id, "method": "POST", "url": "/v1/embeddings", "body": {"model": "text-embedding-3-small", "input": text}})
                jsonlfile.write(jsonl_entry + '\n')
                
                metadata[id] = {
                    "text": text,
                    **{emotion: bool(int(row[emotion])) for emotion in emotion_columns}
                }
                entry_count += 1

    with open(metadata_output, 'w', encoding='utf-8') as metafile:
        json.dump(metadata, metafile)

    return entry_count

def save_batch_info(csv_file: str, jsonl_file: str, metadata_file: str, openai_file_id: str):
    """Save batch information to a JSON file."""
    batch_info = {
        "csv_file": csv_file,
        "jsonl_file": jsonl_file,
        "metadata_file": metadata_file,
        "openai_file_id": openai_file_id
    }
    with open(f"batch_processing/{os.path.basename(csv_file)[:-4]}_batch_info.json", "w") as f:
        json.dump(batch_info, f)

def main(args):
    if args.check_all_status:
        with open("batch_ids.json", "r") as f:
            all_batch_ids = json.load(f)
        for batch_id in all_batch_ids:
            status, errors = check_batch(batch_id)
            output = f"Batch {batch_id} status: {status}"
            if errors:
                output += f" with errors: {errors}"
            print(output)
    elif args.check_status:
        batch_id = args.check_status
        status, errors = check_batch(batch_id)
        output = f"Batch {batch_id} status: {status}"
        if errors:
            output += f" with errors: {errors}"
        print(output)
    elif args.generate_embeddings:
        data_dir = "data/full_dataset/"
        batch_ids = []
        for filename in os.listdir(data_dir):
            if filename.endswith(".csv"):
                csv_path = os.path.join(data_dir, filename)
                jsonl_path = f"{filename[:-4]}.jsonl"
                metadata_path = f"{filename[:-4]}_metadata.json"
                entries_processed = process_csv(csv_path, jsonl_path, metadata_path)
                print(f"Processed {entries_processed} entries from {filename}")
                
                with open(jsonl_path, "rb") as file:
                    response = client.files.create(file=file, purpose="batch")
                openai_file_id = response.id
                
                save_batch_info(csv_path, jsonl_path, metadata_path, openai_file_id)
                
                batch_id = start_batch(openai_file_id)
                batch_ids.append(batch_id)
                print(f"Batch {filename} started, id: {batch_id}")
        
        with open("batch_processing/batch_ids.json", "w") as f:
            json.dump(batch_ids, f)
    elif args.fetch_and_upsert:
        with open("batch_processing/batch_ids.json", "r") as f:
            all_batch_ids = json.load(f)
        
        for batch_id in all_batch_ids:
            while check_batch(batch_id)[0] != "finished":
                print(f"Batch {batch_id} status: {check_batch(batch_id)}")
                time.sleep(10)
            embeddings, input_file_id = fetch_embeddings(batch_id)
            
            # Find the corresponding batch info file
            batch_info_files = [f for f in os.listdir('batch_processing') if f.endswith('_batch_info.json')]
            batch_info_file = next(f for f in batch_info_files if json.load(open(os.path.join('batch_processing', f)))['openai_file_id'] == input_file_id)
            batch_info = json.load(open(os.path.join('batch_processing', batch_info_file)))
            
            do_upsert(embeddings, batch_info['metadata_file'])
        print("Fetch and upsert complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process emotions data")
    parser.add_argument("--check-status", type=str, help="Check status of a batch job")
    parser.add_argument("--check-all-status", action="store_true", help="Check status of all batch jobs")
    parser.add_argument("--generate-embeddings", action="store_true", help="Generate embeddings")
    parser.add_argument("--fetch-and-upsert", action="store_true", help="Fetch embeddings and upsert to Pinecone")
    args = parser.parse_args()
    main(args)