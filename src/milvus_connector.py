from src.constants import (
    MILVUS_URI,
    MILVUS_TOKEN,
    COLLECTION_NAME,
    INITIAL_SEARCH_MULTIPLIER,
    OPENAI_API_KEY,
    ENABLE_RERANKING,
    RERANKING_MODEL,
    MILVUS_DATABASE,
)
from typing import List, Dict, Any
import requests
import json
import openai

openai.api_key = OPENAI_API_KEY
client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

# The cpq_life_sciences schema uses a 1536-dimension vector field
EMBEDDING_MODEL = "text-embedding-3-small"

# Optional: ensure collection exists using pymilvus
try:
    from pymilvus import MilvusClient, DataType
except Exception:
    MilvusClient = None
    DataType = None

def ensure_collection_exists() -> None:
    """Create cpq_life_sciences collection with the expected schema if missing.
    Safe to call multiple times.
    """
    if MilvusClient is None:
        return
    try:
        mc = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN, db_name=MILVUS_DATABASE)
        if mc.has_collection(COLLECTION_NAME):
            return
        schema = mc.create_schema(enable_dynamic_field=True)
        # Auto ID primary key
        schema.add_field("primary_key", DataType.INT64, is_primary=True, auto_id=True, description="Auto primary key")
        # Vector field 1536
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1536, description="Embedding vector")
        # Other fields
        schema.add_field("doc_id", DataType.VARCHAR, max_length=256)
        schema.add_field("chunk_id", DataType.INT64)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("source", DataType.VARCHAR, max_length=256)

        index_params = mc.prepare_index_params()
        index_params.add_index("vector", metric_type="COSINE")

        mc.create_collection(COLLECTION_NAME, schema=schema, index_params=index_params)
    except Exception:
        # Non-fatal: if creation fails (e.g., permissions), upserts may still work if it exists
        return


async def rerank_results(query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Rerank search results using OpenAI's GPT model with prompt-based evaluation."""
    if not client or not documents or not ENABLE_RERANKING:
        return documents[:top_k]
    
    try:
        print(f"Reranking {len(documents)} documents using prompt-based evaluation")
        
        # Prepare documents for reranking
        doc_texts = []
        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            # Create a formatted document string for reranking
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            # Format document with metadata context for better semantic matching
            channel_name = metadata.get('channel_name', 'Unknown')
            video_title = metadata.get('video_title', 'Unknown Video')
            doc_str = f"Channel: {channel_name}\nVideo: {video_title}\nContent: {text}"
            doc_texts.append(doc_str)
        
        # Create evaluation prompt
        evaluation_prompt = f"""
You are an expert at evaluating the relevance of documents to a user query. 

User Query: "{query}"

Please evaluate each document below and assign a relevance score from 0.0 to 1.0, where:
- 0.0 = Completely irrelevant
- 0.5 = Somewhat relevant
- 1.0 = Highly relevant

Consider factors like:
- Semantic similarity to the query
- Whether the document directly addresses the query
- Contextual relevance
- Information completeness

Documents to evaluate:

"""
        
        # Add each document to the prompt with a number
        for i, doc_text in enumerate(doc_texts, 1):
            evaluation_prompt += f"\nDocument {i}:\n{doc_text}\n"
        
        evaluation_prompt += f"""

Please respond with ONLY a JSON array of scores, one for each document, in order.
Example format: [0.8, 0.3, 0.9, 0.1, 0.7]

Scores:"""
        
        # Get relevance scores from GPT
        try:
            response = await client.chat.completions.create(
                model=RERANKING_MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise evaluator. Respond only with the JSON array of scores."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                max_tokens=200,
                temperature=0.1  # Low temperature for consistent scoring
            )
            
            # Parse the response to get scores
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON array from response
            import re
            json_match = re.search(r'\[[0-9.,\s]+\]', response_text)
            if json_match:
                scores_text = json_match.group()
                scores = [float(score.strip()) for score in scores_text.strip('[]').split(',')]
            else:
                # Fallback: try to parse the entire response as JSON
                scores = json.loads(response_text)
            
            # Ensure we have the right number of scores
            if len(scores) != len(documents):
                print(f"Warning: Expected {len(documents)} scores, got {len(scores)}")
                # Pad or truncate scores
                if len(scores) < len(documents):
                    scores.extend([0.0] * (len(documents) - len(scores)))
                else:
                    scores = scores[:len(documents)]
            
        except Exception as e:
            print(f"Error getting GPT scores: {e}")
            # Fallback to uniform scores
            scores = [0.5] * len(documents)
        
        # Create list of (score, index) tuples and sort by score
        scored_docs = [(scores[i], i) for i in range(len(documents))]
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return reranked documents
        reranked_docs = []
        for score, idx in scored_docs[:top_k]:
            doc = documents[idx]
            # Add relevance score to metadata for debugging
            if 'rerank_score' not in doc:
                doc['rerank_score'] = score
            reranked_docs.append(doc)
        
        print(f"Reranking completed. Top relevance scores: {[f'{s:.3f}' for s, _ in scored_docs[:3]]}")
        return reranked_docs
        
    except Exception as e:
        print(f"Error in reranking: {e}")
        return documents[:top_k]


async def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI."""
    if not client:
        return []
    try:
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []

async def search_similar_documents(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for similar documents using Zilliz dedicated API over HTTP with reranking."""        
    try:
        # Get query embedding
        query_embedding = await get_embedding(query)
        print('embeeding', query_embedding)
        if not query_embedding:
            return []

        # Prepare search request for Zilliz API
        search_url = f"{MILVUS_URI}/v2/vectordb/entities/search"
        headers = {
            "Authorization": f"Bearer {MILVUS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Search for more documents initially to allow reranking to select the best ones
        initial_limit = min(limit * INITIAL_SEARCH_MULTIPLIER, 20)  # Get 3x more results for reranking
        
        search_data = {
            "dbName": MILVUS_DATABASE,
            "databaseName": MILVUS_DATABASE,
            "collectionName": COLLECTION_NAME,
            "data": [query_embedding],
            "limit": initial_limit,
            "outputFields": ["text", "doc_id", "chunk_id", "source"]
        }

        response = requests.post(search_url, json=search_data, headers=headers)
        if response.status_code != 200:
            print(f"Zilliz API error: {response.status_code}")
            return []
        
        result = response.json()
        pretty_json_string = json.dumps(result, indent=4)
        print('milvus sources', pretty_json_string)

        sources = []
        if 'data' in result:
            for hit in result['data']:
                try:
                    sources.append({
                        "text": hit.get('text', ''),
                        "metadata": {
                            "doc_id": hit.get('doc_id'),
                            "chunk_id": hit.get('chunk_id'),
                            "source": hit.get('source')
                        }
                    })
                except Exception:
                    sources.append({
                        "text": hit.get('text', ''),
                        "metadata": {}
                    })
        
        # Apply reranking to improve result relevance
        reranked_sources = await rerank_results(query, sources, limit)
        
        pretty_json_string = json.dumps(reranked_sources, indent=4)
        print('reranked sources', pretty_json_string)
    

        return reranked_sources
           
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []



async def add_document(text: str, doc_id: str, chunk_id: int, source: str):
    """Add a document chunk to the cpq_life_sciences collection using Zilliz HTTP API.

    The target schema includes fields: vector (1536), doc_id (varchar), chunk_id (int64), text (varchar), source (varchar).
    Primary key is auto-generated; we do not set it here.
    """
    try:
        # Check if the chunk already exists by doc_id and chunk_id
        query_url = f"{MILVUS_URI}/v2/vectordb/entities/search"
        headers = {
            "Authorization": f"Bearer {MILVUS_TOKEN}",
            "Content-Type": "application/json"
        }

        query_data = {
            "dbName": MILVUS_DATABASE,
            "databaseName": MILVUS_DATABASE,
            "collectionName": COLLECTION_NAME,
            "filter": f"doc_id == \"{doc_id}\" && chunk_id == {chunk_id}",
            "outputFields": ["doc_id", "chunk_id"]
        }

        response = requests.post(query_url, json=query_data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            if result.get('data') and len(result['data']) > 0:
                return {"message": "Document already exists", "doc_id": doc_id, "chunk_id": chunk_id}

        # Get embedding (1536-dim)
        embedding = await get_embedding(text)

        # Upsert into Zilliz using HTTP API
        insert_url = f"{MILVUS_URI}/v2/vectordb/entities/upsert"

        insert_data = {
            "dbName": MILVUS_DATABASE,
            "databaseName": MILVUS_DATABASE,
            "collectionName": COLLECTION_NAME,
            "data": [
                {
                    "vector": embedding,
                    "text": text,
                    "doc_id": doc_id,
                    "chunk_id": int(chunk_id),
                    "source": source,
                }
            ]
        }

        response = requests.post(insert_url, json=insert_data, headers=headers)
        if response.status_code != 200:
            try:
                return {"message": "Failed to insert document", "doc_id": doc_id, "chunk_id": chunk_id, "status": response.status_code, "error": response.text}
            except Exception:
                return {"message": "Failed to insert document", "doc_id": doc_id, "chunk_id": chunk_id, "status": response.status_code}

        try:
            body = response.json()
        except Exception:
            body = None

        return {"message": "Document added successfully", "doc_id": doc_id, "chunk_id": chunk_id, "response": body}

    except Exception:
        return {"message": "Failed to insert document", "doc_id": doc_id, "chunk_id": chunk_id}

        