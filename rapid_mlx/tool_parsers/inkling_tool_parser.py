# SPDX-License-Identifier: Apache-2.0
"""Inkling JSON-block adapter; the wrapper's args become OpenAI arguments."""

import json
import uuid

from ..reasoning.inkling_parser import (
    TOOL,
    InklingBlocks,
    _aligned_token_prefix,
    _token_text_matches,
)
from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


class InklingToolParser(ToolParser):
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("inkling_native",)

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self.reset()

    def reset(self):
        super().reset()
        self.blocks = InklingBlocks(self.model_tokenizer)
        self.payload = None
        self.index = 0
        self.streaming = False
        self.pending_deltas = []

    @staticmethod
    def _call(raw):
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("name"), str):
                return None
            if not parsed["name"]:
                return None
            args = next(
                (parsed[k] for k in ("args", "arguments", "parameters") if k in parsed),
                {},
            )
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                return None
            return {
                "id": "call_" + uuid.uuid4().hex[:24],
                "name": parsed["name"],
                "arguments": json.dumps(args, ensure_ascii=False),
            }
        except (ValueError, TypeError):
            return None

    def _consume(self, events, final=False):
        content, calls = [], []

        def close():
            if self.payload is not None:
                call = self._call(self.payload)
                if call:
                    calls.append(call)
                    if self.streaming:
                        self._stream_call(call)
                else:
                    content.append(self.payload)
                self.payload = None

        for event in events:
            kind, value = event[:2]
            if kind == "marker":
                close()
                if value == TOOL:
                    self.payload = ""
            elif kind == "tool" and self.payload is not None:
                self.payload += value
            elif kind == "content":
                content.append(value)
        if final:
            close()
        return "".join(content), calls

    def extract_tool_calls(self, model_output, request=None):
        parser = type(self)(self.model_tokenizer)
        content, calls = parser._consume(
            parser.blocks.feed(model_output, final=True), final=True
        )
        return ExtractedToolCallInformation(bool(calls), calls, content or None)

    def extract_tool_calls_with_tokens(self, model_output, request=None, *, token_ids):
        if not _token_text_matches(self.model_tokenizer, model_output, token_ids):
            return self.extract_tool_calls(model_output, request)
        parser = type(self)(self.model_tokenizer)
        if not parser.blocks.marker_ids:
            return self.extract_tool_calls(model_output, request)
        content, calls = parser._consume(
            parser.blocks.feed_tokens(token_ids, final=True), final=True
        )
        return ExtractedToolCallInformation(bool(calls), calls, content or None)

    def _stream_call(self, call):
        # Wait for the entire block to validate: a valid argument prefix can
        # still end in malformed JSON, and emitted calls cannot be retracted.
        self.pending_deltas.append(
            {
                "index": self.index,
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
        )
        self.index += 1

    def _wire(self, content):
        deltas, self.pending_deltas = self.pending_deltas, []
        result = {}
        if content:
            result["content"] = content
        if deltas:
            result["tool_calls"] = deltas
            result["preserve_post_tool_content"] = True
        return result or None

    def extract_tool_calls_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids=None,
        current_token_ids=None,
        delta_token_ids=None,
        request=None,
    ):
        self.streaming = True
        self.blocks.disable_tokens()
        content, _ = self._consume(self.blocks.feed(delta_text))
        return self._wire(content)

    def extract_tool_calls_streaming_with_tokens(
        self, previous_text, current_text, delta_text, *, request=None, delta_token_ids
    ):
        self.streaming = True
        aligned = _aligned_token_prefix(
            self.model_tokenizer,
            delta_text,
            self.blocks._deferred_token_ids + list(delta_token_ids),
        )
        if not self.blocks.marker_ids or self.blocks.token_disabled or aligned is None:
            self.blocks.disable_tokens()
            events = self.blocks.feed(delta_text)
        else:
            complete_ids, deferred_ids = aligned
            self.blocks._deferred_token_ids = list(deferred_ids)
            events = self.blocks.feed_tokens(complete_ids)
            if events is None:
                self.blocks.disable_tokens()
                events = self.blocks.feed(delta_text)
        content, _ = self._consume(events)
        return self._wire(content)

    def finalize_legacy_raw_stream(self, full_text, request=None):
        if self.blocks.token_mode:
            content, _ = self._consume(self.blocks.feed_tokens([], final=True) or [])
        else:
            content = ""
        tail, _ = self._consume(self.blocks.feed("", final=True), final=True)
        content += tail
        return self._wire(content)


ToolParserManager.register_module("inkling", InklingToolParser)
