# MCP Resilience and Graceful Degradation

## Overview

The Simple RAG Writer implements comprehensive resilience mechanisms to ensure the system remains usable even when MCP (Model Context Protocol) servers are unavailable, misconfigured, or experiencing transient failures.

## Server Criticality Levels

MCP servers can be marked with three criticality levels in your configuration:

### Required

- **Behavior**: System will not start if server is unavailable
- **User experience**: Prompted to continue or abort on health check failure
- **Use case**: Core functionality servers that are essential for operation
- **Example**: Legal research server for a legal document writer

```yaml
mcp_servers:
  - id: "legal_research"
    command: ["legal-mcp-server"]
    criticality: "required"
```

### Optional (default)

- **Behavior**: System starts even if unavailable
- **User experience**: Warnings shown when server fails, degraded experience
- **Caching**: Results cached and stale cache used as fallback
- **Use case**: Enhancement servers that improve experience but aren't essential
- **Example**: Notes server, papers server, web search

```yaml
mcp_servers:
  - id: "notes"
    command: ["mcp-notes-server"]
    criticality: "optional"  # This is the default
```

### Best Effort

- **Behavior**: Silent failure on unavailability
- **User experience**: No warnings to user
- **Caching**: Cached results used, but failures don't interrupt workflow
- **Use case**: Experimental or low-value servers
- **Example**: Experimental summarization server

```yaml
mcp_servers:
  - id: "experimental_summarizer"
    command: ["experimental-mcp"]
    criticality: "best_effort"
```

## Fallback Mechanisms

The system provides multiple layers of fallback when MCP servers fail:

### 1. Result Caching

- **Duration**: Recent MCP results cached for 1 hour (configurable)
- **Location**: `logs/mcp_cache/` directory
- **Format**: JSON files with hash-based names
- **Behavior**:
  - Fresh cache returned immediately for optional/best_effort servers
  - Successful results automatically cached
  - Cache checked before each call for efficiency

### 2. Stale Cache Fallback

- **Trigger**: When MCP call fails after retries
- **Behavior**: System attempts to use expired (stale) cache results
- **User experience**: Better to have outdated results than no results
- **Logging**: Warnings logged when stale cache is used (when logging enabled)

### 3. Manual Input (`/paste` command)

When MCP servers are unavailable, users can manually provide context:

```
> /paste Meeting notes from yesterday
Paste or type context for 'Meeting notes from yesterday'.
End with a line containing only '###' or press Ctrl+D

[User pastes content here]
###

✓ Added 15 lines of manual context as 'Meeting notes from yesterday'.
```

The pasted content is added to the context buffer and used in subsequent LLM prompts.

### 4. URL Fallback (`/url` command)

Users can fetch web content directly as an alternative to MCP-based web search:

```
> /url https://example.com/article
```

## Health Checking

### Startup Health Check

Before planning mode starts, the system checks all required servers:

1. Lists tools on each server (fast health check)
2. Measures response time
3. If required servers unavailable:
   - Displays table with server status
   - Prompts user: "Continue anyway? (y/N)"
   - Aborts if user declines

### Runtime Diagnostics (`/mcp-status`)

Users can check MCP server health at any time during planning:

```
> /mcp-status
```

Output includes:
- Server availability (✓ or ✗)
- Tool count
- Response time
- Error messages for unavailable servers
- Configuration (timeout, retry settings, command)

## Configuration Example

```yaml
mcp_servers:
  # Core research functionality - must be available
  - id: "papers"
    command: ["mcp-papers-server"]
    criticality: "required"
    timeout: 30
    retry_attempts: 3
    retry_delay_seconds: 2.0

  # Nice to have - degrade gracefully
  - id: "notes"
    command: ["mcp-notes-server"]
    criticality: "optional"
    timeout: 30
    retry_attempts: 2

  # Experimental - fail silently
  - id: "experimental"
    command: ["experimental-mcp"]
    criticality: "best_effort"
    timeout: 10
    retry_attempts: 1
```

## Retry Logic

All MCP tool calls include automatic retry for transient failures:

### Retryable Errors

- Timeouts
- Connection errors (refused, closed, unavailable)
- Server unavailable
- Server failed to start

### Non-Retryable Errors

- Validation errors (bad parameters)
- Tool not found
- Invalid schema

### Retry Behavior

1. First attempt fails with retryable error
2. Wait `retry_delay_seconds`
3. Retry up to `retry_attempts` times
4. If all attempts fail:
   - For required servers: raise error
   - For optional/best_effort servers: try stale cache, then fail gracefully

## User Experience During Degradation

### Required Server Unavailable

```
⚠ Warning: Some required MCP servers are unavailable:

┌────────────┬────────────┬───────────────┬─────────────────────┐
│ Server     │ Critical   │ Status        │ Error               │
├────────────┼────────────┼───────────────┼─────────────────────┤
│ papers     │ Required   │ ✗ Unavailable │ Connection refused  │
└────────────┴────────────┴───────────────┴─────────────────────┘

Continue anyway? (y/N):
```

### Optional Server Failure (with stale cache)

- No interruption to workflow
- Stale cached result used silently
- Debug logs record the fallback (if logging enabled)
- User sees slightly outdated results but can continue working

### Optional Server Failure (no cache)

- Warning displayed: "Optional MCP reference 'notes' failed: Connection timeout"
- Workflow continues with reduced context
- Suggestion to use `/paste` or `/url` as alternative

### Best Effort Server Failure

- Completely silent
- No warnings or interruptions
- Debug logs record the failure (if logging enabled)

## Recommendations

### For Development

- Use `optional` for most servers during development
- Enable debug mode to see cache behavior: `debug_mode: true`
- Test with servers stopped to verify graceful degradation

### For Production

- Mark only truly essential servers as `required`
- Use `optional` for enhancement features
- Monitor cache hit rates in logs
- Set appropriate timeouts (30s default is reasonable)
- Configure retry attempts based on server reliability

### For Experimentation

- Use `best_effort` for experimental or unstable servers
- Prevents experimental features from disrupting workflow
- Easy to promote to `optional` once stable

## Troubleshooting

### Server Always Unavailable at Startup

1. Check `/mcp-status` for detailed error
2. Verify server command is correct
3. Test server independently: `<command> --help`
4. Check logs in `logs/srw_debug.log` (if logging enabled)
5. Increase timeout if server is slow to start

### Stale Cache Used Too Frequently

- Indicates server reliability issues
- Check server logs for patterns
- Consider increasing retry attempts
- May need to restart/fix underlying server

### Context Incomplete After Server Failure

- Use `/paste` to manually add missing context
- Use `/url` to fetch web content directly
- Check `/context` to see what context is active
- Use `/inject` to add previously fetched references

## Implementation Notes

### Cache Directory

- Location: `logs/mcp_cache/`
- Format: `<hash>.json` where hash is SHA256 of (server, tool, params)
- TTL: 3600 seconds (1 hour)
- Cleanup: Automatic on expired cache reads

### Health Check Performance

- Tool listing is fast (<500ms typically)
- Health checks run sequentially at startup
- Only required servers block startup
- Optional/best_effort servers checked but don't block

### Memory Usage

- Cache files are small (typically <100KB each)
- Old cache files automatically deleted on access
- Manual cleanup: `rm -rf logs/mcp_cache/`

## Future Enhancements

Potential improvements for future versions:

- Configurable cache TTL per server
- Cache warming on startup
- Health check dashboard in planning mode
- Metrics export (Prometheus, etc.)
- Automatic server restart on repeated failures
- Circuit breaker pattern for failing servers
