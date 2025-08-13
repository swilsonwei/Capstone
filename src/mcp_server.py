from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi_mcp import FastApiMCP
from typing import Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from src.milvus_connector import search_similar_documents, add_document, ensure_collection_exists
from src.sow_generator import generate_sow
import asyncio
from pathlib import Path
from src.constants import mcp_config
from mcp_use import MCPClient, MCPAgent
from langchain_openai import ChatOpenAI
from src.order_store import create_order, list_orders, update_status, get_order, update_items
from src.audit_log import append_log, list_logs

try:
    from docx import Document
except Exception:
    Document = None
from io import BytesIO
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# Optional PDF generation dependency
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
except Exception:
    canvas = None

app = FastAPI()

class AddDocumentRequest(BaseModel):
    text: str = Field(..., description="Document text to add")
    metadata: dict = Field(default_factory=dict, description="Optional metadata for the document")


@app.post('/v1/search', operation_id="search_for_documents")
async def search(query: str) -> list:
    """Search for documents in Milvus."""
    return await search_similar_documents(query)


@app.post('/v1/add-documents', operation_id="add_documents_to_milvus")
async def add_document_to_milvus(request: AddDocumentRequest) -> Dict:
    """Add a document to Milvus vector database."""
    try:
        print(f"Adding document: {request.text[:100]}...")
        print(f"Metadata: {request.metadata}")
        
        result = await add_document(request.text, request.metadata)
        
        return {
            "success": True,
            "message": "Document added successfully to Milvus",
            "result": result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to add document to Milvus: {str(e)}",
            "text": request.text,
            "metadata": request.metadata
        }

# Define a simple tool
@app.get("/v1/greet")#, operation_id="greet") commenting out makes this mcp server toolkit unexposed
async def greet(name: str) -> Dict:
    """Greets the given name."""
    return {"message": f"Hello, {name}!"}

@app.get("/v2/greet", operation_id="greet")
async def greet(name: str) -> Dict:
    """Greets the given name."""
    return {"message": f"Howdy, {name}!"}

# Define a regular FastAPI endpoint (optional)
@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI MCP server!"}

# Initialize FastAPIMCP
# You can specify include_routes or exclude_routes to control exposed endpoints
mcp_server = FastApiMCP(app,    
    name="MCP API",
    description="MCP server for the API",
    describe_full_response_schema=True,  # Describe the full response JSON-schema instead of just a response example
    describe_all_responses=True,  # Describe all the possible responses instead of just the success (2XX) response
)

# Mount the MCP server to your FastAPI app
mcp_server.mount()

# Mount static files for a simple frontend UI
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.post("/sow/upload")
async def sow_upload(
    file: UploadFile = File(...),
    customer: Optional[str] = Form(default=None),
    title: Optional[str] = Form(default=None),
):
    """Accept a PDF or DOCX, extract text, generate a SOW, and return it."""
    content = await file.read()
    text = ""
    try:
        if file.filename.lower().endswith(".docx"):
            if not Document:
                return {"error": "python-docx not installed"}
            doc = Document(BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())
        elif file.filename.lower().endswith(".pdf"):
            if not PdfReader:
                return {"error": "pypdf not installed"}
            reader = PdfReader(BytesIO(content))
            pages = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t.strip():
                    pages.append(t)
            text = "\n".join(pages)
        else:
            return {"error": "Unsupported file type. Upload .pdf or .docx"}
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

    if not text.strip():
        return {"error": "No extractable text found in the file"}

    sow = await generate_sow(text, customer_name=customer, project_title=title)
    return {"sow": sow}


@app.get("/sow")
async def sow_form():
    return {
        "html": """
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <title>SOW Generator</title>
  </head>
  <body>
    <h1>SOW Generator</h1>
    <form id=\"form\" method=\"post\" action=\"/sow/upload\" enctype=\"multipart/form-data\">
      <label>Customer (optional): <input type=\"text\" name=\"customer\" /></label><br/>
      <label>Title (optional): <input type=\"text\" name=\"title\" /></label><br/>
      <input type=\"file\" name=\"file\" accept=\".pdf,.docx\" required />
      <button type=\"submit\">Generate SOW</button>
    </form>
  </body>
  </html>
        """
    }


class AgentRunRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to run through the MCP agent")


@app.post("/agent/run")
async def agent_run(req: AgentRunRequest):
    try:
        client = MCPClient(mcp_config)
        llm = ChatOpenAI(model="gpt-4o-mini", streaming=False)
        agent = MCPAgent(llm=llm, client=client, max_steps=20)
        result = await agent.run(req.prompt)
        append_log({
            "type": "agent_run",
            "prompt": req.prompt,
            "result": result,
        })
        return {"result": result}
    except Exception as e:
        append_log({"type": "agent_run_error", "prompt": req.prompt, "error": str(e)})
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/agent")
async def agent_page():
    index_path = static_dir / "cpq_agent.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return JSONResponse(status_code=404, content={"error": "agent.html not found"})


@app.get("/rfps")
async def rfps_page():
    page = static_dir / "rfps.html"
    if page.exists():
        return FileResponse(str(page), media_type="text/html")
    return JSONResponse(status_code=404, content={"error": "rfps.html not found"})


@app.get("/orders")
async def orders_page():
    page = static_dir / "orders.html"
    if page.exists():
        return FileResponse(str(page), media_type="text/html")
    return JSONResponse(status_code=404, content={"error": "orders.html not found"})


def _chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


@app.post("/agent/upload")
async def agent_upload(file: UploadFile = File(...)):
    """Upload .pdf or .docx, chunk, and insert into Milvus."""
    content = await file.read()
    text = ""
    try:
        if file.filename.lower().endswith(".docx"):
            if not Document:
                return {"error": "python-docx not installed"}
            doc = Document(BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())
        elif file.filename.lower().endswith(".pdf"):
            if not PdfReader:
                return {"error": "pypdf not installed"}
            reader = PdfReader(BytesIO(content))
            pages = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t.strip():
                    pages.append(t)
            text = "\n".join(pages)
        else:
            return {"error": "Unsupported file type. Upload .pdf or .docx"}
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

    if not text.strip():
        return {"error": "No extractable text found in the file"}

    try:
        ensure_collection_exists()
    except Exception:
        pass

    chunks = _chunk_text(text)
    doc_id = Path(file.filename).stem
    source = file.filename
    tasks = [add_document(chunk, doc_id, idx, source) for idx, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success = sum(1 for r in results if isinstance(r, dict) and r.get("message", "").lower().startswith("document"))

    # Extract quote-like line items and compute subtotal
    items = await extract_line_items(text)
    subtotal = 0.0
    normalized_items = []
    for it in items:
        name = str(it.get("item", "")).strip()
        try:
            qty = float(it.get("quantity", 0))
        except Exception:
            qty = 0.0
        try:
            unit_cost = float(it.get("unit_cost", it.get("cost", 0)))
        except Exception:
            unit_cost = 0.0
        line_total = qty * unit_cost
        subtotal += line_total
        normalized_items.append({
            "item": name,
            "quantity": qty,
            "unit_cost": unit_cost,
            "line_total": line_total,
        })

    order = create_order(file.filename, subtotal, len(normalized_items), status="Quoted", items=normalized_items)
    append_log({
        "type": "upload_ingest",
        "file": file.filename,
        "chunks": len(chunks),
        "inserted": success,
        "order_id": order.get("id"),
        "items_count": len(normalized_items),
        "subtotal": subtotal,
    })
    return {
        "file": file.filename,
        "chunks": len(chunks),
        "inserted": success,
        "items": normalized_items,
        "subtotal": subtotal,
        "order": order,
        "pricing_url": f"/pricing/{order.get('id')}"
    }


async def extract_line_items(text: str):
    """Extract structured line items from text using the OpenAI client.

    Returns: List[{item, quantity, unit_cost}] (best-effort)
    """
    from src.milvus_connector import client as openai_client
    system_msg = (
        "Extract a list of quote line items from the text. "
        "Respond ONLY with JSON in the form: {\"items\":[{\"item\":str,\"quantity\":number,\"unit_cost\":number}, ...]} . "
        "If values are missing, infer reasonable quantities and set unit_cost to 0."
    )
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": text[:20000]},
            ],
            temperature=0.2,
            max_tokens=800,
            stream=False,
        )
        content = resp.choices[0].message.content.strip()
        import json as pyjson, re
        try:
            data = pyjson.loads(content)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", content)
            data = pyjson.loads(m.group(0)) if m else {"items": []}
        items = data.get("items", [])
        return items if isinstance(items, list) else []
    except Exception:
        return []


@app.post("/agent/quote_pdf")
async def agent_quote_pdf(payload: Dict):
    """Generate a PDF from provided items and subtotal; return inline for a new tab."""
    if canvas is None:
        return JSONResponse(status_code=400, content={"error": "reportlab not installed"})
    items = payload.get("items", [])
    try:
        subtotal = float(payload.get("subtotal", 0))
    except Exception:
        subtotal = 0.0

    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter
    y = height - 1 * inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, y, "Quote Preview")
    y -= 0.4 * inch
    c.setFont("Helvetica", 10)
    c.drawString(1 * inch, y, "Item")
    c.drawString(4.2 * inch, y, "Qty")
    c.drawString(4.8 * inch, y, "Unit Cost")
    c.drawString(5.8 * inch, y, "Line Total")
    y -= 0.25 * inch

    c.setFont("Helvetica", 9)
    for it in items:
        if y < 1 * inch:
            c.showPage()
            y = height - 1 * inch
            c.setFont("Helvetica", 9)
        name = str(it.get("item", ""))[:60]
        try:
            qty = float(it.get("quantity", 0))
        except Exception:
            qty = 0.0
        try:
            unit_cost = float(it.get("unit_cost", 0))
        except Exception:
            unit_cost = 0.0
        line_total = float(it.get("line_total", qty * unit_cost))
        c.drawString(1 * inch, y, name)
        c.drawRightString(4.6 * inch, y, f"{qty:g}")
        c.drawRightString(5.6 * inch, y, f"${unit_cost:,.2f}")
        c.drawRightString(7.5 * inch, y, f"${line_total:,.2f}")
        y -= 0.22 * inch

    y -= 0.2 * inch
    c.setFont("Helvetica-Bold", 11)
    c.drawRightString(7.5 * inch, y, f"Subtotal: ${subtotal:,.2f}")

    c.showPage()
    c.save()
    pdf_buffer.seek(0)
    headers = {
        "Content-Disposition": "inline; filename=quote_preview.pdf",
    }
    append_log({"type": "quote_pdf", "items": len(items), "subtotal": subtotal})
    return StreamingResponse(pdf_buffer, media_type="application/pdf", headers=headers)


@app.get("/orders/pdf/{order_id}")
async def orders_pdf(order_id: str):
    if canvas is None:
        return JSONResponse(status_code=400, content={"error": "reportlab not installed"})
    order = get_order(order_id)
    if not order:
        return JSONResponse(status_code=404, content={"error": "order not found"})
    items = order.get("items", [])
    subtotal = float(order.get("subtotal", 0))

    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter
    y = height - 1 * inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, y, f"Quote {order_id}")
    y -= 0.4 * inch
    c.setFont("Helvetica", 10)
    c.drawString(1 * inch, y, f"Source: {order.get('source_file','')}")
    y -= 0.25 * inch
    c.drawString(1 * inch, y, "Item")
    c.drawString(4.2 * inch, y, "Qty")
    c.drawString(4.8 * inch, y, "Unit Cost")
    c.drawString(5.8 * inch, y, "Line Total")
    y -= 0.25 * inch

    c.setFont("Helvetica", 9)
    for it in items:
        if y < 1 * inch:
            c.showPage()
            y = height - 1 * inch
            c.setFont("Helvetica", 9)
        name = str(it.get("item", ""))[:60]
        try:
            qty = float(it.get("quantity", 0))
        except Exception:
            qty = 0.0
        try:
            unit_cost = float(it.get("unit_cost", 0))
        except Exception:
            unit_cost = 0.0
        line_total = float(it.get("line_total", qty * unit_cost))
        c.drawString(1 * inch, y, name)
        c.drawRightString(4.6 * inch, y, f"{qty:g}")
        c.drawRightString(5.6 * inch, y, f"${unit_cost:,.2f}")
        c.drawRightString(7.5 * inch, y, f"${line_total:,.2f}")
        y -= 0.22 * inch

    y -= 0.2 * inch
    c.setFont("Helvetica-Bold", 11)
    c.drawRightString(7.5 * inch, y, f"Subtotal: ${subtotal:,.2f}")

    c.showPage()
    c.save()
    pdf_buffer.seek(0)
    headers = {"Content-Disposition": f"inline; filename={order_id}.pdf"}
    return StreamingResponse(pdf_buffer, media_type="application/pdf", headers=headers)


@app.get("/logs/data")
async def logs_data(limit: int = 200):
    return {"logs": list_logs(limit=limit)}


class PricingUpdate(BaseModel):
    items: list


@app.get("/pricing/{order_id}")
async def pricing_page(order_id: str):
    page = static_dir / "pricing.html"
    if page.exists():
        return FileResponse(str(page), media_type="text/html")
    return JSONResponse(status_code=404, content={"error": "pricing.html not found"})


@app.get("/pricing/data/{order_id}")
async def pricing_data(order_id: str):
    order = get_order(order_id)
    if not order:
        return JSONResponse(status_code=404, content={"error": "order not found"})
    return {
        "order": {
            "id": order.get("id"),
            "source_file": order.get("source_file"),
            "items": order.get("items", []),
            "subtotal": order.get("subtotal", 0),
            "status": order.get("status")
        }
    }


@app.post("/pricing/save/{order_id}")
async def pricing_save(order_id: str, body: PricingUpdate):
    updated = update_items(order_id, body.items or [])
    if not updated:
        return JSONResponse(status_code=404, content={"error": "order not found"})
    append_log({"type": "pricing_save", "order_id": order_id, "items": len(updated.get("items", [])), "subtotal": updated.get("subtotal", 0)})
    return {"ok": True, "order": updated}


@app.get("/orders/data")
async def orders_data():
    return {"orders": list_orders()}


class OrderStatusUpdate(BaseModel):
    id: str
    status: str


@app.post("/orders/status")
async def orders_status(update: OrderStatusUpdate):
    if update.status not in ("Quoted", "Sent", "Received"):
        return JSONResponse(status_code=400, content={"error": "invalid status"})
    updated = update_status(update.id, update.status)
    if not updated:
        return JSONResponse(status_code=404, content={"error": "order not found"})
    return updated


@app.get("/orders/stats")
async def orders_stats():
    orders = list_orders()
    total_cost = sum(float(o.get("subtotal", 0) or 0) for o in orders)
    open_count = sum(1 for o in orders if (o.get("status") in ("Quoted", "Sent")))
    return {
        "total_cost": total_cost,
        "open_count": open_count,
        "total_count": len(orders)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 