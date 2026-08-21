"""Provider-neutral text and image content contracts."""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

ImageMediaType: TypeAlias = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
SUPPORTED_IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})


@dataclass(frozen=True)
class TextBlock:
    """One immutable text content block."""

    text: str


@dataclass(frozen=True, repr=False)
class ImageBlock:
    """One immutable local image content block."""

    data: bytes
    media_type: ImageMediaType

    def __repr__(self) -> str:
        """Return a representation that never includes image data."""
        return f"ImageBlock(media_type={self.media_type!r}, size_bytes={len(self.data)})"


ContentBlock: TypeAlias = TextBlock | ImageBlock
Prompt: TypeAlias = str | Sequence[ContentBlock]
NormalizedContent: TypeAlias = tuple[ContentBlock, ...]


def normalize_content(value: Prompt | Any, *, label: str = "content") -> NormalizedContent:
    """Copy and validate public text/image content."""
    if isinstance(value, str):
        blocks: tuple[Any, ...] = (TextBlock(value),)
    elif isinstance(value, Sequence):
        blocks = tuple(value)
    else:
        raise TypeError(f"{label} must be a string or a sequence of content blocks")
    if not blocks:
        raise ValueError(f"{label} must not be empty")
    normalized: list[ContentBlock] = []
    for index, block in enumerate(blocks):
        if isinstance(block, TextBlock):
            if not isinstance(block.text, str) or not block.text:
                raise ValueError(f"{label} text block {index} must not be empty")
            normalized.append(block)
            continue
        if isinstance(block, ImageBlock):
            if type(block.data) is not bytes:
                raise TypeError(f"{label} image block {index} data must be bytes")
            if not block.data:
                raise ValueError(f"{label} image block {index} data must not be empty")
            if block.media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
                raise ValueError(f"{label} image block {index} has unsupported media type: {block.media_type!r}")
            normalized.append(block)
            continue
        raise TypeError(f"{label} block {index} has unsupported type: {type(block).__name__}")
    return tuple(normalized)


def content_to_json(blocks: Sequence[ContentBlock]) -> list[dict[str, Any]]:
    """Encode validated blocks into the canonical JSON shape."""
    normalized = normalize_content(blocks)
    encoded: list[dict[str, Any]] = []
    for block in normalized:
        if isinstance(block, TextBlock):
            encoded.append({"type": "text", "text": block.text})
        else:
            encoded.append({
                "type": "image",
                "media_type": block.media_type,
                "data": base64.b64encode(block.data).decode("ascii"),
            })
    return encoded


def content_from_json(value: Any, *, label: str = "content") -> NormalizedContent:
    """Strictly decode canonical JSON content blocks."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array")
    blocks: list[ContentBlock] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{label} block {index} must be an object")
        if item.get("type") == "text":
            if set(item) != {"type", "text"} or not isinstance(item.get("text"), str):
                raise ValueError(f"{label} text block {index} has wrong shape")
            blocks.append(TextBlock(item["text"]))
            continue
        if item.get("type") == "image":
            if set(item) != {"type", "media_type", "data"}:
                raise ValueError(f"{label} image block {index} has wrong shape")
            media_type = item.get("media_type")
            data = item.get("data")
            if not isinstance(media_type, str) or not isinstance(data, str):
                raise ValueError(f"{label} image block {index} has wrong type")
            try:
                decoded = base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"{label} image block {index} has invalid base64 data") from exc
            blocks.append(ImageBlock(decoded, media_type))  # type: ignore[arg-type]
            continue
        raise ValueError(f"{label} block {index} has unsupported type")
    return normalize_content(blocks, label=label)


def contains_image(blocks: Sequence[ContentBlock]) -> bool:
    """Return whether content includes an image block."""
    return any(isinstance(block, ImageBlock) for block in blocks)


def append_text_block(blocks: Sequence[ContentBlock], text: str) -> NormalizedContent:
    """Append non-empty text while preserving existing block boundaries."""
    if not text:
        return normalize_content(blocks)
    return normalize_content((*blocks, TextBlock(text)))


def text_only_value(blocks: Sequence[ContentBlock]) -> str | None:
    """Join all-text content with the public provider-boundary separator."""
    normalized = normalize_content(blocks)
    if contains_image(normalized):
        return None
    return "\n\n".join(block.text for block in normalized if isinstance(block, TextBlock))


def redacted_content_json(blocks: Sequence[ContentBlock]) -> list[dict[str, Any]]:
    """Project blocks without image bytes while preserving order."""
    normalized = normalize_content(blocks)
    projected: list[dict[str, Any]] = []
    for index, block in enumerate(normalized):
        if isinstance(block, TextBlock):
            projected.append({"type": "text", "text": block.text})
        else:
            projected.append({
                "type": "image",
                "media_type": block.media_type,
                "size_bytes": len(block.data),
                "block_index": index,
            })
    return projected


def redacted_content_string(blocks: Sequence[ContentBlock]) -> str:
    """Return compact redacted JSON for a multimodal event field."""
    return json.dumps(redacted_content_json(blocks), ensure_ascii=False, separators=(",", ":"))


_DATA_URL_RE = re.compile(r"data:image/[^;,\s]+;base64,[A-Za-z0-9+/=]+", re.IGNORECASE)


def redact_image_data(text: str, blocks: Sequence[ContentBlock] = ()) -> str:
    """Redact complete image encodings from an error message."""
    redacted = _DATA_URL_RE.sub("[image data redacted]", text)
    for block in blocks:
        if not isinstance(block, ImageBlock):
            continue
        encoded = base64.b64encode(block.data).decode("ascii")
        redacted = redacted.replace(encoded, "[image data redacted]")
        redacted = redacted.replace(f"data:{block.media_type};base64,{encoded}", "[image data redacted]")
    return redacted
