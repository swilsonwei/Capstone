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

    print("\n--- Testing Local Server Tools ---")
    result = await agent.run("if my workflow includes flow cytometry and it's a toxicology study, tell me approximatelyhow much I could quote my customer.")
    print(f"Agent result: {result}")

    # GitHub MCP functionality removed for this project

if __name__ == "__main__":
    asyncio.run(main())