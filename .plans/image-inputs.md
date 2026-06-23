# Image Inputs

ThinHarness should support images by making model input richer while keeping plain text prompts unchanged. The public API should continue to accept `Harness.run("prompt")`, and additionally accept an ordered sequence of input parts such as text plus image content.

The core type should be provider-neutral:

```python
UserInput = str | Sequence[UserInputPart]
UserInputPart = TextPart | ImageUrlPart | BinaryImagePart
```

`TextPart` can stay optional at first because bare strings inside the sequence are enough for most callers. `ImageUrlPart` should hold a URL, optional MIME type, optional provider metadata, and an optional `force_download` flag. `BinaryImagePart` should hold bytes, MIME type, and optional provider metadata, with helpers such as `from_path(...)` and `from_data_uri(...)`.

The harness should normalize prompt handling once near the run boundary. Hooks, tracing, stream events, and model session APIs should receive the neutral input shape instead of assuming a string. Text-only callers should see the same behavior and payloads they see today.

Provider adapters should own wire-format mapping:

- OpenAI Responses: map text to `input_text` and images to `input_image`; binary images should use data URIs, URL images should use URLs unless `force_download` is set.
- Anthropic Messages: map text to text blocks and images to image blocks; binary images should use base64 source blocks, URL images should use URL source blocks when supported.
- OpenRouter chat completions: map text and images to chat content parts using `text` and `image_url`; binary images should use data URIs.

Unsupported image cases should fail loudly with a provider error. The harness should not silently stringify images or drop them.

Keep the first implementation limited to images. Do not include audio, video, PDFs, uploaded provider files, or prompt-cache markers yet. The type shape should leave room for those later, but the code should not implement speculative modalities.

Tests should cover backward-compatible text prompts, mixed text/image ordering, URL image mapping, binary image mapping, provider-specific unsupported cases, notice appending with rich input, and tracing redaction or placeholder behavior so raw image bytes are not accidentally written into local traces.
