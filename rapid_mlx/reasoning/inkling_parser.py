# SPDX-License-Identifier: Apache-2.0
"""Typed-content state machine adapted to Rapid parser lifecycle interfaces."""

from .base import DeltaMessage, ReasoningParser

MESSAGE = "<|message_model|>"
THINKING = "<|content_thinking|>"
TEXT = "<|content_text|>"
TOOL = "<|content_invoke_tool_json|>"
END = "<|end_message|>"
SAMPLING_END = "<|content_model_end_sampling|>"
MARKERS = (
    MESSAGE,
    THINKING,
    TEXT,
    TOOL,
    END,
    SAMPLING_END,
    "<|content_invoke_tool_text|>",
    "<|content_tool_error|>",
)


def _token_text_matches(tokenizer, text, token_ids):
    if tokenizer is None:
        return False
    try:
        return tokenizer.decode(list(token_ids), skip_special_tokens=False) == text
    except (AttributeError, TypeError, ValueError, UnicodeError):
        return False


def _aligned_token_prefix(tokenizer, text, token_ids):
    """Split a stable delta from trailing IDs held for incomplete UTF-8."""
    if _token_text_matches(tokenizer, text, token_ids):
        return tuple(token_ids), ()
    if tokenizer is None:
        return None
    try:
        for split in range(max(0, len(token_ids) - 8), len(token_ids) + 1):
            prefix = tokenizer.decode(
                list(token_ids[:split]), skip_special_tokens=False
            )
            if prefix != text:
                continue
            tail = tokenizer.decode(list(token_ids[split:]), skip_special_tokens=False)
            if tail and all(char == "\ufffd" for char in tail):
                return tuple(token_ids[:split]), tuple(token_ids[split:])
    except (AttributeError, TypeError, ValueError, UnicodeError):
        return None
    return None


class InklingBlocks:
    """Incremental typed-block transitions, shared by both parser passes.

    Header text is metadata. Thinking, text and JSON blocks may repeat in
    any order; either end marker closes a block. The text grammar also works
    on reconstructed content passed from reasoning to the tool parser.
    """

    def __init__(self, tokenizer=None):
        self.state = "content"
        self.buffer = ""
        self.tool_quoted = False
        self.tool_escaped = False
        self.tokenizer = tokenizer
        self.marker_ids = self._marker_ids(tokenizer)
        self._deferred_token_ids = []
        self.token_mode = False
        self.token_disabled = False

    @staticmethod
    def _marker_ids(tokenizer):
        if tokenizer is None:
            return {}
        try:
            vocab = tokenizer.get_vocab()
            ids = {vocab[marker]: marker for marker in MARKERS if marker in vocab}
        except (AttributeError, KeyError, TypeError, NotImplementedError):
            return {}
        return ids if len(ids) == len(MARKERS) else {}

    def disable_tokens(self):
        """Fall back permanently for this stream after text/ID divergence."""
        self.token_mode = False
        self.token_disabled = True
        self._deferred_token_ids.clear()

    def feed_tokens(self, token_ids, final=False):
        """Parse IDs as framing; ordinary token text is never marker-scanned."""
        if not self.marker_ids or self.token_disabled:
            return None
        self.token_mode = True
        events = []
        ids = list(token_ids)
        index = 0
        while index < len(ids):
            marker = self.marker_ids.get(ids[index])
            if marker is None:
                end = index + 1
                while end < len(ids) and ids[end] not in self.marker_ids:
                    end += 1
                run = ids[index:end]
                text = self.tokenizer.decode(run, skip_special_tokens=False)
                if self.state == "tool":
                    self._update_quote_state(text)
                events.append((self.state, text, tuple(run)))
                index = end
                continue
            token_id = ids[index]
            if marker == MESSAGE:
                self.state = "header"
            elif marker == THINKING:
                self.state = "reasoning"
            elif marker == TOOL:
                self.state = "tool"
                self.tool_quoted = False
                self.tool_escaped = False
            elif marker in (END, SAMPLING_END):
                self.state = "content"
            else:
                self.state = "content"
            events.append(("marker", marker, (token_id,)))
            index += 1
        if final:
            self._deferred_token_ids.clear()
        return events

    def _update_quote_state(self, text):
        for char in text:
            if self.tool_escaped:
                self.tool_escaped = False
            elif self.tool_quoted and char == "\\":
                self.tool_escaped = True
            elif char == '"':
                self.tool_quoted = not self.tool_quoted

    def feed(self, text, final=False):
        self.buffer += text
        events = []
        while self.buffer:
            framing = self.state != "tool" or not self.tool_quoted
            marker = (
                next((m for m in MARKERS if self.buffer.startswith(m)), None)
                if framing
                else None
            )
            if marker:
                self.buffer = self.buffer[len(marker) :]
                if marker == MESSAGE:
                    self.state = "header"
                elif marker == THINKING:
                    self.state = "reasoning"
                elif marker == TOOL:
                    self.state = "tool"
                    self.tool_quoted = False
                    self.tool_escaped = False
                elif marker in (END, SAMPLING_END):
                    self.state = "content"
                else:
                    self.state = "content"
                events.append(("marker", marker))
                continue
            if (
                framing
                and not final
                and any(m.startswith(self.buffer) for m in MARKERS)
            ):
                break
            end = 1
            while end < len(self.buffer) and self.buffer[end] != "<":
                end += 1
            value = self.buffer[:end]
            if self.state == "tool":
                # Carry string/escape state across deltas, including a delta
                # ending with a backslash immediately before a quote.
                self._update_quote_state(value)
            events.append((self.state, value))
            self.buffer = self.buffer[end:]
        return events


class InklingReasoningParser(ReasoningParser):
    # Typed channels remain authoritative when a request disables thinking.
    # Keep parsing before the standard stream sanitizer strips their markers.
    sanitize_when_thinking_disabled = True

    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(tokenizer)
        self.reset_state()

    @property
    def reasoning_start_str(self):
        return THINKING

    @property
    def reasoning_end_str(self):
        return END

    def reset_state(self):
        self.blocks = InklingBlocks(self.tokenizer)
        self.tool = False
        self.tool_view_noncontent = False
        self.reasoning_whitespace = ""

    def adjust_request(self, request):
        request.skip_special_tokens = False
        return request

    def _delta(self, events):
        segments = []
        content_token_ids = []
        has_token_ids = False
        for event in events:
            kind, value = event[:2]
            ids = event[2] if len(event) > 2 else None
            has_token_ids |= ids is not None
            if kind == "marker":
                if value != THINKING:
                    self.reasoning_whitespace = ""
                # Preserve any boundary closing a tool block so the second
                # parser pass can validate it before consuming the next block.
                if self.tool or self.tool_view_noncontent or value == TOOL:
                    segments.append(("content", value))
                    if ids:
                        content_token_ids.extend(ids)
                    # If the forwarded closer enters a header or reasoning
                    # block, also forward its exit: its body is stripped here,
                    # but the tool pass still needs to return to content.
                    self.tool_view_noncontent = value in (MESSAGE, THINKING)
                self.tool = value == TOOL
                continue
            if kind == "reasoning":
                value = self.reasoning_whitespace + value
                trimmed = value.rstrip()
                self.reasoning_whitespace = value[len(trimmed) :]
                if trimmed:
                    segments.append(("reasoning", trimmed))
            elif kind in ("content", "tool"):
                segments.append(("content", value))
                if ids:
                    content_token_ids.extend(ids)
        reasoning = "".join(v for k, v in segments if k == "reasoning")
        content = "".join(v for k, v in segments if k == "content")
        return (
            DeltaMessage(
                reasoning=reasoning or None,
                content=content or None,
                source_segments=tuple(segments),
                content_token_ids=(tuple(content_token_ids) if has_token_ids else None),
            )
            if segments
            else None
        )

    def extract_reasoning(self, model_output, enable_thinking=None):
        parser = type(self)(self.tokenizer)
        msg = parser._delta(parser.blocks.feed(model_output, final=True))
        return (msg.reasoning, msg.content) if msg else (None, None)

    def extract_reasoning_with_tokens(
        self, model_output, *, token_ids, enable_thinking=None
    ):
        if not _token_text_matches(self.tokenizer, model_output, token_ids):
            return self.extract_reasoning(model_output, enable_thinking)
        blocks = InklingBlocks(self.tokenizer)
        if not blocks.marker_ids:
            return self.extract_reasoning(model_output, enable_thinking)
        parser = type(self)(self.tokenizer)
        msg = parser._delta(blocks.feed_tokens(token_ids, final=True))
        return (msg.reasoning, msg.content) if msg else (None, None)

    def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
        self.blocks.disable_tokens()
        return self._delta(self.blocks.feed(delta_text))

    def extract_reasoning_streaming_with_tokens(
        self, previous_text, current_text, delta_text, *, delta_token_ids
    ):
        aligned = _aligned_token_prefix(
            self.tokenizer,
            delta_text,
            self.blocks._deferred_token_ids + list(delta_token_ids),
        )
        if not self.blocks.marker_ids or self.blocks.token_disabled or aligned is None:
            self.blocks.disable_tokens()
            return self._delta(self.blocks.feed(delta_text))
        complete_ids, deferred_ids = aligned
        self.blocks._deferred_token_ids = list(deferred_ids)
        events = self.blocks.feed_tokens(complete_ids)
        if events is None:
            self.blocks.disable_tokens()
            events = self.blocks.feed(delta_text)
        return self._delta(events)

    def finish_stream(self):
        if self.blocks.token_mode:
            return self._delta(self.blocks.feed_tokens([], final=True) or [])
        return self._delta(self.blocks.feed("", final=True))

    def is_open_in_think(self, accumulated_text):
        blocks = InklingBlocks()
        blocks.feed(accumulated_text, final=True)
        return blocks.state == "reasoning"
