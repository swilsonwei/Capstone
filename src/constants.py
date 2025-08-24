import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()
GITHUB_MCP_PAT = os.getenv("GITHUB_PAT")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_URI = os.getenv("MILVUS_URI", "https://in03-745f9d0d214e290.serverless.gcp-us-west1.cloud.zilliz.com")
# Do not ship a default token; require env configuration
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
COLLECTION_NAME = "cpq_life_sciences"
MILVUS_DATABASE = os.getenv("MILVUS_DATABASE", "default")
MILVUS_ENABLED = os.getenv("MILVUS_ENABLED", "false").lower() == "true"

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# Clerk configuration
CLERK_PUBLISHABLE_KEY = (
    os.getenv("NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY")
    or os.getenv("CLERK_PUBLISHABLE_KEY", "")
)
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY", "")
CLERK_FRONTEND_API = os.getenv("CLERK_FRONTEND_API", "")
CLERK_API_URL = os.getenv("CLERK_API_URL", "https://api.clerk.com")
CLERK_JWKS_URL = os.getenv(
    "CLERK_JWKS_URL",
    "https://ample-wren-6.clerk.accounts.dev/.well-known/jwks.json",
)

# Internal API secret to allow MCP-to-API calls to bypass Clerk auth (server-to-server)
INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET", "")

# Reranking configuration
# Default reranking off for speed unless explicitly enabled
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "false").lower() == "true"
RERANKING_MODEL = os.getenv("RERANKING_MODEL", "gpt-4o-mini")  # Chat model used for prompt-based reranking
INITIAL_SEARCH_MULTIPLIER = int(os.getenv("INITIAL_SEARCH_MULTIPLIER", "3"))  # How many more results to fetch initially

# Abuse-protection / RAG safety configuration
ENABLE_SAFETY_FILTERS = os.getenv("ENABLE_SAFETY_FILTERS", "true").lower() == "true"
BLOCK_ON_INJECTION = os.getenv("BLOCK_ON_INJECTION", "true").lower() == "true"
ENABLE_OPENAI_MODERATION = os.getenv("ENABLE_OPENAI_MODERATION", "false").lower() == "true"
MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", "600"))
MAX_DOC_CHARS = int(os.getenv("MAX_DOC_CHARS", "4000"))

# Simple per-IP rate limiting (token bucket)
RATE_LIMIT_RPS = float(os.getenv("RATE_LIMIT_RPS", "3"))
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "6"))


mcp_config = {
    "mcpServers": {
        "local-mcp-server": {
            "type": "http",
            # Use Render's port if present; defaults to 8000 locally
            "url": f"http://localhost:{os.getenv('PORT', '8000')}/mcp",
            "headers": {
                "Content-Type": "application/json",
            }
        }
    }
}

# Optionally add GitHub MCP server if token is provided
if GITHUB_MCP_PAT:
    mcp_config["mcpServers"]["github-mcp-remote"] = {
        "type": "http",
        "url": "https://api.githubcopilot.com/mcp/",
        "headers": {"Authorization": f"Bearer {GITHUB_MCP_PAT}"},
    }

# Attach internal secret header if configured
if INTERNAL_API_SECRET:
    try:
        mcp_config["mcpServers"]["local-mcp-server"]["headers"]["X-Internal-Secret"] = INTERNAL_API_SECRET
    except Exception:
        pass
