# Plugin foundation review packet

Review these two implementation plans as one ordered migration:

1. [Plugin system and filesystem plugin](37-plugin-system-and-filesystem.md)
2. [MCP plugin](38-mcp-plugin.md)

Read both linked files in full. Review each plan and the seam between them. In particular, check whether plan 37 leaves a complete and testable plugin interface, whether plan 38 can use that interface without MCP-specific changes to core, and whether the temporary subagent bridges create avoidable dual systems.

Return one verdict with findings labeled `Plan 37`, `Plan 38`, or `Cross-plan`. Raise decisions that require product-owner input separately from implementation corrections.
