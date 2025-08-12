from typing import List, Dict
import json

from constants import RERANKING_MODEL
from milvus_connector import search_similar_documents, client


async def generate_sow(upload_text: str, customer_name: str | None = None, project_title: str | None = None) -> str:
    """Generate a Statement of Work using uploaded notes and retrieved context."""
    related_docs: List[Dict] = await search_similar_documents(upload_text, limit=8)

    context_blocks: List[str] = []
    for doc in related_docs:
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        try:
            md = metadata if isinstance(metadata, dict) else json.loads(str(metadata))
        except Exception:
            md = {}
        source = md.get("source") or md.get("file_name") or "unknown"
        context_blocks.append(f"Source: {source}\n{text}")

    context_snippets = "\n\n---\n\n".join(context_blocks[:5])

    title_line = project_title or "Generated Statement of Work"
    customer_line = f"Customer: {customer_name}" if customer_name else ""

    prompt = f"""
You are a contracts specialist. Draft a clear, concise Statement of Work (SOW) based on the user's notes and the retrieved context.

Include these sections with headings:
- Title
- Customer
- Background
- Scope
- Deliverables
- Acceptance Criteria
- Timeline
- Assumptions
- Out of Scope
- Pricing (placeholder)
- Terms (placeholder)

User Notes:
"""
    prompt += upload_text[:20000]
    prompt += "\n\nRetrieved Context (for reference, cite inline as [Source: <name>] where useful):\n"
    prompt += context_snippets[:20000]

    system_msg = "You write precise, unambiguous SOWs. Keep it professional and structured."

    resp = await client.chat.completions.create(
        model=RERANKING_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Title: {title_line}\n{customer_line}\n\n" + prompt},
        ],
        temperature=0.4,
        max_tokens=1200,
        stream=False,
    )

    return resp.choices[0].message.content.strip()


