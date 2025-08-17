import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()
GITHUB_MCP_PAT = os.getenv("GITHUB_PAT")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_URI = os.getenv("MILVUS_URI", "https://in03-745f9d0d214e290.serverless.gcp-us-west1.cloud.zilliz.com")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "a9fbb8b5142250c0449cbda360b72567b993a860ada6fe23c2a5b7af1cadce39b41eae18491af928622d13593ec6391c9b067efc")
COLLECTION_NAME = "cpq_life_sciences"
MILVUS_DATABASE = os.getenv("MILVUS_DATABASE", "default")

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
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
RERANKING_MODEL = os.getenv("RERANKING_MODEL", "gpt-4o-mini")  # Chat model used for prompt-based reranking
INITIAL_SEARCH_MULTIPLIER = int(os.getenv("INITIAL_SEARCH_MULTIPLIER", "3"))  # How many more results to fetch initially


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
