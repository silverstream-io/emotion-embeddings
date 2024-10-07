# Emotion Embeddings

This project processes a dataset of text comments with associated emotions, generates embeddings using OpenAI's text-embedding model, and stores them in a Pinecone vector database for efficient similarity search and analysis.

## Dataset

The dataset used in this project is derived from the GoEmotions dataset, which can be found at:
https://github.com/google-research/google-research/blob/master/goemotions/README.md

**Disclaimer**: This project uses the GoEmotions dataset created by Google Research. We do not claim ownership of the dataset and use it in accordance with its license. Please refer to the original source for more information about the dataset and its terms of use.

## Project Overview

This project accomplishes the following tasks:

1. Reads CSV files containing text comments and associated emotion labels.
2. Generates embeddings for each text comment using OpenAI's 'text-embedding-3-small' model.
3. Stores the embeddings along with metadata in a Pinecone vector database.

The metadata includes:

- The original text
- Boolean values for each emotion category

## Setup

1. Clone this repository.
2. Install the required dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `PINECONE_API_KEY`: Your Pinecone API key
   - `PINECONE_INDEX`: The name of your Pinecone index

## Usage

To generate embeddings:

```bash
python src/process_emotions.py --generate-embeddings
```

To check the status of a batch job:

```bash
python src/process_emotions.py --check-status <batch_id>
```

To check the status of all batch jobs:

```bash
python src/process_emotions.py --check-all-status
```

To retrieve and save embeddings:

```bash
python src/process_emotions.py --retrieve-and-save
```

To upsert embeddings:

```bash
python src/process_emotions.py --upsert
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
