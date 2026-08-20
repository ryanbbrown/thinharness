# ThinHarness image support plan

Status: Revised after review panel v1. No implementation is authorized by this plan.

## Goal

Add first-class image input without expanding ThinHarness into a general media runtime.

ThinHarness will support ordered text and image content in:

- user prompts
- custom tool results
- the filesystem plugin through an opt-in image-reading tool
- successful MCP tool results

OpenAI, Anthropic, and OpenRouter models will receive the same neutral content through provider-specific wire formats. Resume remains self-contained and portable across built-in providers.

## Non-goals

- image generation or assistant-produced images
- audio, video, files, or realtime voice
- remote image URLs or URL fetching
- image editing, resizing, OCR, or format conversion
- automatic image loading from Bash output
- model-name tables that predict vision support
- token-by-token streaming
- image-bearing subagent tasks or `parallel_llm` prompts; child task inputs stay text-only

## Product decisions

### One neutral content seam

Add a small leaf module, `thinharness/content.py`, with immutable content blocks:

```python
@dataclass(frozen=True)
class TextBlock:
    text: str

@dataclass(frozen=True)
class ImageBlock:
    data: bytes
    media_type: Literal[
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
    ]

ContentBlock = TextBlock | ImageBlock
Prompt = str | Sequence[ContentBlock]
```

Normalize a string prompt to one `TextBlock` at the public interface. Special-case `str` before sequence handling. Copy caller sequences to tuples on entry and preserve block order.

Use immutable `bytes` plus an explicit media type as the only public image form. Do not accept `bytearray`, `memoryview`, paths, URLs, or provider-native image dictionaries. Provider adapters encode base64. Give `ImageBlock` a redacted `repr` that never prints image data.

Reject empty string prompts, empty block sequences, empty text blocks, empty image data, unsupported media types, and unknown block values before the first provider request. Empty strings are accepted today, so this is an explicit behavior change. Do not inspect model names for vision support. A provider or custom model reports that a selected model cannot process images.

### Prompt behavior

`Harness.run()`, `stream()`, and `run_sync()` accept `Prompt`. Approval resume does not accept new user content.

`RunStartContext.prompt` and `UserPromptSubmitContext.prompt` change from `str` to normalized content-block tuples. A hook may replace either field with a string or a valid block sequence; the harness normalizes and validates the replacement after each hook phase. `additional_context` stays `list[str]` and appends one final `TextBlock`. Harness notices remain separate notice transcript entries and append after hook context. The order sent to a provider is:

1. caller blocks
2. hook context
3. harness notices

At the provider boundary, an all-text user entry joins its blocks with `\n\n`. This keeps the existing string payload for a normal string prompt plus hook context. A mixed image entry preserves every block boundary and order.

Rename the public custom-session method `continue_with_user_text` to `continue_with_user_content`. Update all unit and end-to-end fakes. This is a pre-1.0 interface change and avoids keeping a misleading name.

### Tool-result behavior

Allow `ToolResult.content` to be either a string or an ordered sequence of `ContentBlock` values. A bare block sequence returned by a handler is also accepted and becomes `ToolResult(ok=True, content=...)`. Normalize before provider serialization. Existing text-only tools continue to expose string content to callers.

A normalized multimodal tool result keeps:

- `ok` as the execution outcome
- ordered text and image content
- JSON metadata, including retry guidance, visible to the model

Keep the current text-only provider output byte-for-byte: one JSON string with `ok`, string `content`, and `metadata`. For an image-bearing result, OpenAI and Anthropic receive a content array with:

1. one text block containing compact JSON for `{"ok": ..., "metadata": ...}`
2. each `ToolResult.content` block in its original order, mapped to a provider text or image block

The first block is the multimodal envelope header. It keeps execution state and retry metadata visible without putting image bytes in JSON text. Provider output preserves the tool-call ID and block order.

`ToolOutput`, tool-call records, and v4 resume entries carry the normalized `ToolResult`, not only a serialized output string. The canonical JSON codec represents image data exactly once.

Keep `AfterToolCallContext.original_output` and `output` as canonical JSON strings, including base64 image fields, and keep `envelope` as the structured mutation interface. Mutating either `output` or `envelope` updates the other and runs strict canonical validation. Document these hook fields as sensitive and potentially large. Remove every `json.dumps(..., default=str)` path that could stringify an unknown block.

`BashPlugin` remains text-only. A command that prints an image path returns that path as text. Bash never reads or infers an image.

### Provider mappings

Keep existing text-only payload shapes unchanged. Use multimodal arrays only when content contains an image.

#### OpenAI Responses

- User text maps to `input_text`.
- User images map to `input_image` with a base64 data URL.
- Image-bearing tool results use the documented `function_call_output.output` content array. The envelope header and tool text map to `input_text`; images map to `input_image`.
- This array support is confirmed by the official [OpenAI function-calling guide](https://developers.openai.com/api/docs/guides/function-calling). Keep one live contract journey because SDK and model behavior can still differ.

#### Anthropic Messages

- Text maps to a `text` block.
- Images map to base64 `image` source blocks.
- Tool-result text and images stay inside the matching `tool_result.content` block list.

#### OpenRouter Chat Completions

OpenRouter documentation is not consistent about image parts in `role="tool"` messages. Use a portable follow-up user-message projection instead of depending on that shape:

- User text and images map to `text` and `image_url` content parts.
- Each normal tool message keeps its `tool_call_id` and contains one canonical JSON string. For image-bearing content, image data becomes a descriptor with media type, byte size, and block index; `ok`, text content, and metadata remain in this tool message.
- After all matching tool messages in a parallel batch, add one `role="user"` message. For each tool image in tool-result order, add a label text part, `[tool image call_id=<id> block=<index>]`, immediately followed by its `image_url` part.
- The labelled user message is a provider wire projection only. The durable transcript keeps the neutral tool result, so live and resumed rendering use the same rule.
- A harness notice follows the labelled image parts in the same user message.

The OpenRouter live journey must verify this projection with a current vision-capable model.

### Resume state

Change user transcript content to canonical content-block arrays. Change each tool transcript entry from an output string to `call_id` plus canonical `ok`, `content`, and `metadata` fields. Encode images in resume JSON as:

```json
{
  "type": "image",
  "media_type": "image/png",
  "data": "<base64>"
}
```

Text blocks use `{"type": "text", "text": "..."}`.

Bump the built-in transcript resume version from 3 to 4. Reject older versions with the existing regenerate-state guidance. Keep the approval envelope version unchanged because it already contains independently versioned provider state. An approval envelope captured before this change fails on resume with `approval state provider_state version 3 is not supported`; test and document this release break.

Resume stays self-contained. Image bytes therefore appear as base64 and add about 33% encoding overhead. Document resume state and completed results as sensitive and potentially large.

Cross-provider resume must preserve block order and image bytes. Same-provider resume must continue to preserve native reasoning data.

### Tracing and events

Never put image bytes, base64, or data URLs into trace attributes or non-terminal progress events, including when text message capture is enabled.

Project each image as a descriptor:

```json
{
  "type": "image",
  "media_type": "image/png",
  "size_bytes": 12345,
  "block_index": 1
}
```

Keep text blocks visible under the existing capture settings. Preserve content order and add each image block's zero-based index to event and trace projections.

Keep current event field types and text-only values unchanged. For multimodal values, `RunStartedEvent.prompt` and `ToolCallCompletedEvent.output` contain compact redacted JSON strings. `ToolCallCompletedEvent.message` stays a plain text summary and never receives a block sequence. Apply the same projection before writing `gen_ai.prompt` and `gen_ai.tool.call.result` trace attributes. Cover `core.py`, `tool_execution.py`, subagent tracing, and transcript rendering as well as the projection and event modules.

`RunCompletedEvent.result` remains the same complete result returned by `run()`. It can contain image-bearing resume state and tool-call records. Document this terminal object as sensitive instead of weakening result identity.

Before creating a provider error event or span error, replace each run-known full base64 value and data URL with `[image data redacted]`. Also redact any complete `data:image/...;base64,...` value found by pattern. Provider adapters must not include serialized request bodies in error messages.

### Filesystem plugin

Add an opt-in `read_image` tool. Do not change the text `read` tool and do not add `read_image` to the filesystem plugin's default tool set.

`read_image` will:

- reuse the filesystem plugin's read path policy
- read one local file as bounded bytes
- detect PNG, JPEG, GIF, or WebP from file signatures
- return compact metadata as one `TextBlock`, followed by one `ImageBlock`
- reject directories, missing files, path escapes, unknown formats, SVG, and oversized files

Add a frozen `max_image_bytes` filesystem-plugin setting with a default of `5_000_000`. Require a positive integer. This limit is independent of `max_read_bytes`; exactly the limit is accepted and one byte over is rejected. Do not add Pillow or another image dependency. Validate the full required signature and minimum header length, not only the first magic bytes. The tool validates container format only; providers validate dimensions and model-specific limits.

`read_image` returns `ToolResult(ok=True, content=[TextBlock(metadata), ImageBlock(...)], metadata=...)`. The compact metadata text comes first and includes the path, media type, and byte size.

### MCP plugin

Preserve successful MCP image blocks as `ImageBlock` values in their original order. If `structuredContent` is present, it remains the authoritative text as required by current behavior: emit its canonical JSON as the first `TextBlock`, discard MCP text blocks, then append image blocks and image placeholders from `result.content` in their original relative order.

A supported media type with valid base64 becomes an `ImageBlock`. Malformed base64 and unsupported image media types become the existing `[image: <mime>]` text placeholder instead of failing the tool call. Continue converting audio, embedded resources, and resource links to text placeholders.

Protocol-level MCP errors remain text-only retry results.

## Behavior documentation

After plan approval and before implementation, update only affected sections of `docs/behavior.md`:

- add a new image-content section covering public blocks, validation, ordering, provider behavior, tool results, and exclusions
- update Resume State for transcript version 4 and self-contained image data
- update Model Observability Projections for redacted image descriptors
- update Filesystem Plugin for opt-in `read_image` and `max_image_bytes`
- update MCP Client Layer so successful image blocks remain images
- update hook and stream requirements where they currently assume string prompts or outputs

## Implementation sequence

### 1. Add and test neutral content types

Create `thinharness/content.py` with public types, normalization, validation, canonical JSON encoding, canonical JSON decoding, and redacted descriptor helpers.

Export public content types from `thinharness/__init__.py`.

Prove that string prompts and text-only tool results retain their current observable behavior.

### 2. Move the run and hook interfaces to content

Update:

- `thinharness/core.py`
- `thinharness/hooks.py`
- `thinharness/turns.py`
- `thinharness/runtime.py`
- `thinharness/events.py`

Pass normalized content from public entry points to `ModelSession`. Append hook context and notices as text blocks without changing image order. Add redacted run-start event and trace projection in this phase, so the type change cannot expose image data.

Update custom-model protocols and test fakes to use `continue_with_user_content`.

### 3. Move tool execution to content

Update:

- `thinharness/tools/base.py`
- `thinharness/tool_execution.py`
- after-tool hook handling
- tool-call records

Keep all existing text tools returning strings. Add multimodal normalization only at the shared tool seam. Define the exact envelope-header serializer and redact tool completion events and trace attributes in this phase.

### 4. Implement provider serializers

Update exact request construction in `thinharness/providers.py` for:

- initial multimodal prompts
- resumed prompts
- mixed text/image tool results
- parallel tool-result batches
- notices added after multimodal content

Keep text-only provider request payloads unchanged. Implement the decided OpenRouter labelled user-message projection in both live tool continuation and transcript replay.

### 5. Upgrade transcript resume

Update transcript entries, codecs, validation, provider transcript renderers, and approval-resume coverage for version 4.

Test same-provider and cross-provider resume with mixed text and images.

### 6. Redact observability surfaces

Audit and complete redaction in:

- `thinharness/core.py`
- `thinharness/tool_execution.py`
- `thinharness/projections.py`
- `thinharness/tracing.py`
- `thinharness/events.py`
- `thinharness/plugins/subagents.py`
- `scripts/build_transcripts.py`

Ensure no trace or non-terminal event contains base64 or data URLs. Earlier phases add redaction at each new image-bearing seam; this phase is the full leak audit.

### 7. Add image-producing built-ins

Add opt-in filesystem `read_image`, then preserve MCP images. Keep Bash unchanged.

### 8. Update documentation and release notes

Update:

- `README.md`
- `docs/docs.md`
- `CHANGELOG.md`
- API exports and examples

Version changes follow the normal release process rather than this implementation plan.

## Tests

Tests must enter through public run, tool, resume, and plugin interfaces. Provider HTTP handlers are the only mocked boundary. Exact expected payloads must be literal fixtures derived from provider formats, not generated by production serializers.

### Content contract

- normalize string prompts to one text block
- preserve mixed block order
- reject empty and unsupported content, including the newly invalid empty string prompt
- detach caller-owned sequences and reject mutable byte containers
- redact `ImageBlock.__repr__`
- keep `thinharness/content.py` independent of core, providers, and tools
- round-trip canonical text and image JSON

### Provider contracts

For OpenAI, Anthropic, and OpenRouter:

- send an initial image prompt
- send interleaved text and images
- continue after a text tool result without changing existing payload shape
- continue after an image tool result while keeping `ok` and metadata model-visible
- preserve several parallel tool results and their call IDs
- keep every existing text-only prompt and tool envelope request body byte-for-byte
- append notices after image content
- migrate both prompt hook fields and normalize valid hook replacements
- surface the normal provider error for a non-vision model

### Resume

- same-provider image replay for all built-in providers
- every cross-provider pair, including replay of a foreign tool image through OpenRouter's labelled projection
- mixed user and tool images
- preserved native reasoning on same-provider replay
- malformed base64, wrong keys, unsupported media types, and old versions fail clearly
- approval pause and resume with image-bearing prior state
- a pre-change approval envelope fails with the documented nested provider-state version error

### Tool behavior

- custom sync and async tools return image content
- image content survives mutation through both after-tool hook fields
- malformed hook JSON and unknown block values fail strict canonical validation
- failed and retryable image results keep `message` text-only and keep retry metadata visible to the model
- failed and retryable tools remain text-only unless explicitly returned otherwise
- parallel mixed text/image tool batches preserve order
- Bash never auto-loads a printed image path

### Filesystem and MCP

- `read_image` is opt-in and absent from defaults
- allowed PNG, JPEG, GIF, and WebP signatures
- truncated headers and files with only spoofed leading magic bytes fail
- exactly `max_image_bytes` succeeds and one byte over fails
- missing file, directory, unreadable file, path escape, and symlink escape
- plugin configuration remains frozen and binding performs no image I/O
- MCP preserves mixed text/image order; audio and resources remain placeholders
- MCP `structuredContent` plus images follows the declared precedence
- malformed MCP base64 and unsupported image media types become placeholders

### Observability

- traces preserve text and image descriptors in order
- traces and progress events contain no image bytes, base64, or data URLs
- provider errors cannot leak known base64 or data URLs through `RunFailedEvent` or span errors
- `RunStartedEvent`, `ToolCallCompletedEvent`, subagent traces, and transcript renderers expose descriptors only
- completed results retain full resume state

### Live validation

Add an opt-in image journey for each built-in provider. Each journey must prove that a real vision-capable model can:

1. answer a question about a supplied image
2. inspect an image returned by a custom tool
3. resume from the resulting state

The OpenRouter journey verifies the labelled user-message image projection. The OpenAI journey verifies the documented `function_call_output.output` content-array contract.

## Validation commands

After implementation:

```bash
uv run ruff check thinharness tests
uv run pytest tests/unit
uv run pyright
```

Run relevant provider journeys when credentials are available. Do not require paid live calls in normal CI.

## Main risks

- **OpenRouter attribution:** The portable labelled user-message projection keeps call IDs visible but cannot give the image a native tool role.
- **Data leakage:** Existing trace, event, hook, and error paths assume text and may expose base64 unless every projection is updated.
- **State growth:** Self-contained resume and tool records can become large.
- **Interface spread:** Prompt and tool content currently use strings across most core seams.
- **False support claims:** A provider protocol can support images while a selected model does not.
- **Text regressions:** Multimodal normalization must not alter current text-only payloads or tool envelopes.

## Review decision

Review panel v1 requested changes. This revision resolves the blocking provider projections, tool envelope, hook mutation, MCP precedence, event redaction, and filesystem limit decisions.

The plan is ready for human approval. After approval, update `docs/behavior.md` with this contract before implementation starts.
