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

def _get_batch_info(batch_id: str) -> Dict[str, Any]:
    """Get the batch info for a batch embedding job."""
    response = client.batches.retrieve(batch_id)
    return response

def check_batch(batch_id: str) -> str:
    """Check the status of a batch embedding job and return a formatted string."""
    response = _get_batch_info(batch_id)
    status = response.status
    errors = response.errors or []
    input_file = response.input_file_id
    output_file = response.output_file_id if response.status == "completed" else "Not available yet"
    
    output = f"Batch {batch_id}:\n"
    output += f"  Status: {status}\n"
    output += f"  Input file: {input_file}\n"
    output += f"  Output file: {output_file}\n"
    if errors:
        output += f"  Errors: {errors}\n"
    return output

def do_upsert(embeddings: List[Dict[str, Any]], metadata_file: str):
    """Upsert a batch of vectors to Pinecone."""
    if not "batch_processing" in metadata_file:
        metadata_file = "batch_processing/" + metadata_file
    if not os.path.exists(metadata_file):
        raise ValueError(f"Metadata file not found: {metadata_file}")
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing metadata file: {metadata_file}")
    except Exception as e:
        raise ValueError(f"Error reading metadata file: {e}")
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

def save_batch_info(batch_id: str, csv_file: str, jsonl_file: str, metadata_file: str, openai_input_file_id: str = "", openai_output_file_id: str = ""):
    """Save batch information to a JSON file."""
    batch_info = {
        "batch_id": batch_id,
        "csv_file": csv_file,
        "jsonl_file": jsonl_file,
        "metadata_file": metadata_file,
        "openai_input_file_id": openai_input_file_id,
        "openai_output_file_id": openai_output_file_id
    }
    with open(f"batch_processing/{os.path.basename(csv_file)[:-4]}_batch_info.json", "w") as f:
        json.dump(batch_info, f)

def retrieve_and_save_embeddings(batch_id: str, output_dir: str):
    """Retrieve embeddings for a batch, save them to a file, and update batch info."""
    print(f"Retrieving embeddings for batch {batch_id}")
    response = client.batches.retrieve(batch_id)
    if response.status not in ["finished", "completed"]:
        print(f"Batch {batch_id} is not ready. Status: {response.status}")
        return

    if not response.output_file_id:
        print(f"No output file ID found for batch {batch_id}")
        return

    output_file = response.output_file_id
    print(f"Downloading content from file: {output_file}")
    content = client.files.content(output_file)
    content_str = content.read().decode('utf-8')

    output_path = os.path.join(output_dir, f"embeddings_{batch_id}.jsonl")
    with open(output_path, 'w') as f:
        f.write(content_str)

    print(f"Embeddings for batch {batch_id} saved to {output_path}")

    # Update batch info file
    batch_info_files = [f for f in os.listdir('batch_processing') if f.endswith('_batch_info.json')]
    batch_info_file = next((f for f in batch_info_files if json.load(open(os.path.join('batch_processing', f)))['openai_file_id'] == batch_id), None)

    if batch_info_file:
        batch_info_path = os.path.join('batch_processing', batch_info_file)
        with open(batch_info_path, 'r+') as f:
            batch_info = json.load(f)
            batch_info['output_file_id'] = output_file
            f.seek(0)
            json.dump(batch_info, f, indent=2)
            f.truncate()
        print(f"Updated batch info file: {batch_info_file} with output file ID")
    else:
        print(f"Warning: Batch info file not found for batch {batch_id}")

def upsert_embeddings(batch_id: str):
    """Upsert embeddings for a batch by combining embedding and metadata files."""
    embeddings_file = f"batch_processing/embeddings/embeddings_{batch_id}.jsonl"
    print(f"Upserting embeddings for batch {batch_id} from {embeddings_file}")
    if not os.path.exists(embeddings_file):
        print(f"Embeddings file not found for batch {batch_id}")
        return

    # Find the corresponding batch info file
    batch_info_files = [f for f in os.listdir('batch_processing') if f.endswith('_batch_info.json')]
    batch_info_file = next((f for f in batch_info_files if json.load(open(os.path.join('batch_processing', f)))['batch_id'] == batch_id), None)

    if not batch_info_file:
        print(f"Batch info file not found for batch {batch_id}")
        return

    batch_info = json.load(open(os.path.join('batch_processing', batch_info_file)))
    metadata_file = batch_info['metadata_file']
    metadata_file = "batch_processing/" + metadata_file

    if not os.path.exists(metadata_file):
        print(f"Metadata file not found: {metadata_file}")
        return

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    vectors = []
    with open(embeddings_file, 'r') as f:
        for line in f:
            emb = json.loads(line)
            id = emb['custom_id'].split('-')[-1]
            if id in metadata:
                vectors.append((emb['custom_id'], emb['response']['body']['data'][0]['embedding'], metadata[id]))
            else:
                print(f"Warning: Metadata not found for {emb['id']}")

    for ids_vectors_chunk in chunks(vectors, batch_size=200):
        index.upsert(vectors=ids_vectors_chunk, namespace='emotions')

    print(f"Upsert complete for batch {batch_id}")

def main(args):
    if args.check_all_status:
        with open("batch_processing/batch_ids.json", "r") as f:
            all_batch_ids = json.load(f)
        for batch_id in all_batch_ids:
            print(check_batch(batch_id))
    elif args.check_status:
        batch_id = args.check_status
        print(check_batch(batch_id))
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
                openai_input_file_id = response.id
                
                batch_id = start_batch(openai_input_file_id)
                batch_info = _get_batch_info(batch_id)
                save_batch_info(batch_id, csv_path, jsonl_path, metadata_path, batch_info.input_file_id, batch_info.output_file_id)
                batch_ids.append(batch_id)
                print(f"Batch {filename} started, id: {batch_id}")
        
        with open("batch_processing/batch_ids.json", "w") as f:
            json.dump(batch_ids, f)
    elif args.retrieve_embeddings:
        print("Starting embedding retrieval process...")
        with open("batch_processing/batch_ids.json", "r") as f:
            all_batch_ids = json.load(f)
        print(f"Loaded {len(all_batch_ids)} batch IDs")

        output_dir = "batch_processing/embeddings"
        os.makedirs(output_dir, exist_ok=True)

        for batch_id in all_batch_ids:
            try:
                retrieve_and_save_embeddings(batch_id, output_dir)
            except Exception as e:
                print(f"Error processing batch {batch_id}: {str(e)}")
                import traceback
                traceback.print_exc()

        print("Embedding retrieval process complete!")
    elif args.upsert:
        print("Starting upsert process...")
        with open("batch_processing/batch_ids.json", "r") as f:
            all_batch_ids = json.load(f)
        print(f"Loaded {len(all_batch_ids)} batch IDs")

        for batch_id in all_batch_ids:
            upsert_embeddings(batch_id)

        print("Upsert process complete!")
    else: 
        print("No arguments provided. Please provide one of the following: --check-status, --check-all-status, --generate-embeddings, --retrieve-embeddings, --upsert")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process emotions data")
    parser.add_argument("--check-status", "-c", type=str, help="Check status of a batch job")
    parser.add_argument("--check-all-status", "-ca", action="store_true", help="Check status of all batch jobs")
    parser.add_argument("--generate-embeddings", "-ge", action="store_true", help="Generate embeddings")
    parser.add_argument("--retrieve-embeddings", "-re", action="store_true", help="Retrieve and save embeddings")
    parser.add_argument("--upsert", "-u", action="store_true", help="Upsert embeddings to Pinecone")
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\nProcess interrupted. Exiting gracefully...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")