from src.constants import (
    MILVUS_URI,
    MILVUS_TOKEN,
    COLLECTION_NAME,
    INITIAL_SEARCH_MULTIPLIER,
    OPENAI_API_KEY,
    ENABLE_RERANKING,
    RERANKING_MODEL,
    MILVUS_DATABASE,
    MILVUS_ENABLED,
    ENABLE_SAFETY_FILTERS,
    BLOCK_ON_INJECTION,
    ENABLE_OPENAI_MODERATION,
    MAX_QUERY_CHARS,
    MAX_DOC_CHARS,
)
from typing import List, Dict, Any
import requests
import json
from collections import OrderedDict
import openai
import re

openai.api_key = OPENAI_API_KEY
client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

# The cpq_life_sciences schema uses a 1536-dimension vector field
EMBEDDING_MODEL = "text-embedding-3-small"
_EMBED_CACHE: OrderedDict[str, list] = OrderedDict()
_EMBED_CACHE_CAPACITY = 512


def _sanitize_text(text: str, max_len: int) -> str:
    try:
        s = (text or "").strip()
        # Normalize whitespace
        s = re.sub(r"\s+", " ", s)
        if len(s) > max_len:
            s = s[:max_len]
        return s
    except Exception:
        return (text or "")[:max_len]


def _looks_like_injection(text: str) -> bool:
    """Heuristic prompt-injection detector for queries and retrieved docs."""
    try:
        s = (text or "").lower()
        if not s:
            return False
        patterns = [
            r"ignore (all|any|previous) instructions",
            r"disregard (the )?rules",
            r"act as (?:a|an)? ",
            r"you are (?:no longer|now) ",
            r"system prompt",
            r"developer message",
            r"override (?:the )?safety",
            r"bypass (?:the )?restrictions",
            r"do not follow (?:the )?guidelines",
            r"exfiltrate (?:the )?prompt",
            r"reveal (?:the )?(?:secrets|keys|internal)",
            r"tool (?:schema|list|instructions)",
        ]
        for pat in patterns:
            if re.search(pat, s):
                return True
        # Excessive control characters or markup
        if s.count("###") > 5 or s.count("```") > 3:
            return True
        return False
    except Exception:
        return False


async def _is_disallowed_via_moderation(text: str) -> bool:
    if not ENABLE_OPENAI_MODERATION or not client:
        return False
    try:
        resp = await client.moderations.create(model="omni-moderation-latest", input=(text or "")[:10000])
        # Support both list and dict shapes across SDK versions
        result = None
        try:
            result = resp.results[0]
        except Exception:
            result = getattr(resp, "result", None)
        if not result:
            return False
        flagged = bool(getattr(result, "flagged", False)) or bool(result.get("flagged", False) if isinstance(result, dict) else False)
        return flagged
    except Exception:
        return False

def _get_cached_embedding(key: str) -> list | None:
    try:
        if key in _EMBED_CACHE:
            # Move to end (most recently used)
            _EMBED_CACHE.move_to_end(key)
            return _EMBED_CACHE[key]
    except Exception:
        pass
    return None

def _put_cached_embedding(key: str, value: list) -> None:
    try:
        if not value:
            return
        _EMBED_CACHE[key] = value
        _EMBED_CACHE.move_to_end(key)
        while len(_EMBED_CACHE) > _EMBED_CACHE_CAPACITY:
            _EMBED_CACHE.popitem(last=False)
    except Exception:
        pass


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
        # Safety: sanitize and optionally block on injection or moderation
        sanitized = _sanitize_text(text, MAX_DOC_CHARS)
        if ENABLE_SAFETY_FILTERS:
            if BLOCK_ON_INJECTION and _looks_like_injection(sanitized):
                print("Blocked embedding due to suspected injection content in text")
                return []
        if await _is_disallowed_via_moderation(sanitized):
            print("Blocked embedding due to moderation policy")
            return []

        cached = _get_cached_embedding(sanitized)
        if cached is not None:
            return cached
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=sanitized
        )
        emb = response.data[0].embedding
        _put_cached_embedding(sanitized, emb)
        return emb
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []

async def search_similar_documents(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for similar documents using Zilliz dedicated API over HTTP with reranking."""        
    try:
        if not MILVUS_ENABLED:
            return []
        # Query safety checks
        safe_query = _sanitize_text(query, MAX_QUERY_CHARS)
        if ENABLE_SAFETY_FILTERS:
            if BLOCK_ON_INJECTION and _looks_like_injection(safe_query):
                print("Blocked search due to suspected prompt injection in query")
                return []
        if await _is_disallowed_via_moderation(safe_query):
            print("Blocked search due to moderation policy")
            return []
        # Get query embedding
        query_embedding = await get_embedding(safe_query)
        # Avoid logging full embedding to reduce noise; log only basic stats
        try:
            emb_len = len(query_embedding) if isinstance(query_embedding, list) else 0
            print(f"Embedding generated: dim={emb_len}")
        except Exception:
            pass
        if not query_embedding:
            return []

        # Prepare search request for Zilliz API
        search_url = f"{MILVUS_URI}/v2/vectordb/entities/search"
        headers = {
            "Authorization": f"Bearer {MILVUS_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # If token is not configured, skip remote call gracefully
        if not MILVUS_TOKEN:
            print("Milvus/Zilliz token not configured; skipping vector search")
            return []
        
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

        response = requests.post(search_url, json=search_data, headers=headers, timeout=5)
        if response.status_code != 200:
            try:
                body = response.text
            except Exception:
                body = ""
            # Truncate long error bodies
            snippet = (body or "")[:240]
            print(f"Zilliz API error: {response.status_code} {snippet}")
            return []
        
        result = response.json()
        # Reduce log volume: log counts only
        try:
            count = len(result.get('data', []) or [])
            print(f"Milvus sources returned: {count}")
        except Exception:
            pass

        sources = []
        if 'data' in result:
            for hit in result['data']:
                try:
                    text_raw = hit.get('text', '')
                    text_clean = _sanitize_text(text_raw, MAX_DOC_CHARS)
                    if ENABLE_SAFETY_FILTERS and _looks_like_injection(text_clean):
                        # Skip suspicious doc chunks
                        continue
                    sources.append({
                        "text": text_clean,
                        "metadata": {
                            "doc_id": hit.get('doc_id'),
                            "chunk_id": hit.get('chunk_id'),
                            "source": hit.get('source')
                        }
                    })
                except Exception:
                    sources.append({
                        "text": _sanitize_text(hit.get('text', ''), MAX_DOC_CHARS),
                        "metadata": {}
                    })
        
        # Apply reranking to improve result relevance
        reranked_sources = await rerank_results(query, sources, limit)
        # Log top texts briefly
        try:
            preview = [s.get('text', '')[:60] for s in reranked_sources[:3]]
            print(f"Reranked top: {preview}")
        except Exception:
            pass
    

        return reranked_sources
           
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []



async def add_document(text: str, doc_id: str, chunk_id: int, source: str):
    """Add a document chunk to the cpq_life_sciences collection using Zilliz HTTP API.

    The target schema includes fields: vector (FLOAT_VECTOR 1536), doc_id (varchar), chunk_id (int64), text (varchar), source (varchar).
    Primary key (INT64) is auto-generated; we do not set it here.
    """
    try:
        if not MILVUS_ENABLED:
            return {"message": "Skipped insert (Milvus disabled)", "doc_id": doc_id, "chunk_id": chunk_id}
        if not MILVUS_TOKEN:
            return {"message": "Skipped insert (no Milvus token)", "doc_id": doc_id, "chunk_id": chunk_id}
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

        response = requests.post(query_url, json=query_data, headers=headers, timeout=5)
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

        response = requests.post(insert_url, json=insert_data, headers=headers, timeout=8)
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

        