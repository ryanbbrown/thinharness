# Decisions

## MCP

- **Resolved server IDs remain server-local.** MCP servers store their resolved ID on the server object. Reusing the same server instance across unrelated harnesses can carry that resolved ID with it, but the v4 plan chose this shape and the edge case is narrow.
- **Content block parsing stays lightweight.** MCP tool result text extraction currently uses the block's `type` value instead of SDK-specific `isinstance` checks. Richer dispatch is deferred until the SDK surface makes it necessary.
- **Serialized MCP config is out of scope.** `HarnessConfig.model_dump()` may include live `MCPServer` objects that are not JSON-roundtrippable. Declarative, serialized MCP configuration can be added as a separate feature.
- **Collision errors remain concise.** Tool name collisions identify the duplicated tool and point users toward `tool_prefix` or `exclude_tools`. Source-specific collision diagnostics are deferred.
- **Lock-held shutdown stays simple.** A rare shared-server timing path can allow a replacement session while the old session finishes tearing down. The current shutdown behavior is accepted instead of adding more lifecycle locking.
- **Coverage changes are separate from MCP.** Pytest coverage configuration and CI coverage dependencies are intentionally committed separately from the MCP implementation.
