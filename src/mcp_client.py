from mcp_use import MCPClient, MCPAgent
from langchain_openai import ChatOpenAI
from src.constants import mcp_config, GITHUB_MCP_PAT
import asyncio
import os 

async def main():
    # Create MCPClient from configuration dictionary
    client = MCPClient(mcp_config) 

    print(f"✅ MCP Client initialized with {len(mcp_config['mcpServers'])} servers:")
    for server_name in mcp_config['mcpServers'].keys():
        print(f"  - {server_name}")

    # Create LLM
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    print("\n--- Generating dynamic quote from uploaded items and Milvus context ---")
    # The CPQ frontend posts uploaded items and embeds to the backend; here we guide the agent
    prompt = (
        "If a user uploads a file, give me a quote that takes in all of the potential line items and cost, "
        "and given the Milvus data/vector embeddings, create a quote/order for the user. "
        "Use the available MCP tools to read context and produce a structured breakdown with totals."
        "If the user asks a question, answer it with the available MCP tools."
    )
    result = await agent.run(prompt)
    print(f"Agent result: {result}")

    # GitHub MCP functionality removed for this project

if __name__ == "__main__":
    asyncio.run(main())