# MCP Tool Servers


## Where to find additional tools

You can find lots of MCP Tool Servers to use with this software at the following websites. It should go without saying that there's probably a lot of redundancy between these:

- [MCPServers.org](https://mcpservers.org/)
- [MCP Servers Awesomelist #1](https://github.com/punkpeye/awesome-mcp-servers)
- [Smithery.ai](https://smithery.ai/)
- [LobeHub MCP Server Directory](https://lobehub.com/mcp)
- [MCP.so](https://mcp.so/)
- [MCP Servers Awesomelist #2](https://github.com/appcypher/awesome-mcp-servers)
- [MCP Market](https://mcpmarket.com/)
- [MCP Servers Awesomelist #3](https://github.com/TensorBlock/awesome-mcp-servers)
- [MCP Server Finder](https://www.mcpserverfinder.com/)
- [MCP Servers Awesomelist #4](https://github.com/jaw9c/awesome-remote-mcp-servers)
- [Cursor's Directory of MCP Servers](https://cursor.directory/mcp)
- [MCP-Cloud.ai](https://mcp-cloud.ai/)
- [Docker's Directory of MCP Servers](https://hub.docker.com/u/mcp)
- [github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)
- [Postman's Directory of MCP Servers](https://www.postman.com/explore/mcp-servers)

## Default MCP Servers

I'm going to try and keep the default set limited only to the MCP Tools that represent a minimal toolset for writing. Depending on the type of writing you're doing, there may be some in here that are superfluous in your use case. You can remove those from your copy of `config.yaml` and skip the installation for those tools.

If nothing else, having these here provides you a way to familiarize yourself with how they work in the context of an LLM based application.

### ARXIV - Scholarly Journal Articles Search / Fulltext

#### Tools List

- **search_papers**:
- **download_paper**:
- **list_papers**:
- **read_paper**:

#### Installation

the best way - and only way I was able to get this working was using [`uv`](https://docs.astral.sh/uv/getting-started/installation) so if you don't have it yet, you'll want to install it... 

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

to install the arxiv mcp server run:

```bash
uv tool install arxiv-mcp-server
```

### Deliberate Reasoning Engine 

#### Tools List

- **log_thought**: Log a structured thought in the reasoning graph
- **get_thought_graph**: Get the current reasoning graph
- **invalidate_assumption**: Mark an assumption as invalid

#### Installation

```bash
npm install -g deliberate-reasoning-engine
```



