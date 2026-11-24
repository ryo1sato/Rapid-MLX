"""Token identity keeps Inkling framing separate from ordinary text."""

import json
from unittest.mock import MagicMock

from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.parsing.token_ids import (
    extract_reasoning_streaming,
    extract_tool_calls_streaming,
)
from rapid_mlx.reasoning import get_parser
from rapid_mlx.service.postprocessor import StreamingPostProcessor
from rapid_mlx.tool_parsers import ToolParserManager
from rapid_mlx.utils.decode import IncrementalDecoder

M = "<|message_model|>"
R = "<|content_thinking|>"
T = "<|content_text|>"
C = "<|content_invoke_tool_json|>"
E = "<|end_message|>"
S = "<|content_model_end_sampling|>"
OTHER = ("<|content_invoke_tool_text|>", "<|content_tool_error|>")
MARKERS = (M, R, T, C, E, S, *OTHER)
BAD = '{"name":"broken","args":{"value":"unterminated'
GOOD = '{"name":"lookup","args":{"query":"ready"}}'


class Tokenizer:
    def __init__(self):
        self.special_ids = {marker: 1000 + i for i, marker in enumerate(MARKERS)}
        self.id_specials = {value: key for key, value in self.special_ids.items()}

    def get_vocab(self):
        return self.special_ids.copy()

    def decode(self, ids, *, skip_special_tokens=False):
        result = []
        byte_run = bytearray()

        def flush():
            if byte_run:
                result.append(bytes(byte_run).decode("utf-8", errors="replace"))
                byte_run.clear()

        for token_id in ids:
            marker = self.id_specials.get(token_id)
            if marker:
                flush()
                if not skip_special_tokens:
                    result.append(marker)
            else:
                byte_run.append(token_id)
        flush()
        return "".join(result)


def _encode(parts):
    tokenizer = Tokenizer()
    ids = []
    for part in parts:
        ids.extend(
            [tokenizer.special_ids[part]]
            if part in tokenizer.special_ids
            else part.encode()
        )
    return tokenizer, ids, tokenizer.decode(ids)


def _scenario(literal_marker=None):
    literal = literal_marker or "ordinary query"
    payload = json.dumps({"name": "lookup", "args": {"query": literal}})
    return _encode(
        [
            M,
            R,
            "Think before.",
            E,
            M,
            C,
            BAD,
            E,
            M,
            R,
            "Think after.",
            E,
            M,
            T,
            "Visible answer.",
            E,
            M,
            C,
            payload,
            E,
            M,
            T,
            "Done é.",
            E,
            S,
        ]
    )


def _complete(tokenizer, ids, text):
    reasoning_parser = get_parser("inkling")(tokenizer)
    tool_parser = ToolParserManager.get_tool_parser("inkling")(tokenizer)
    reasoning, _ = reasoning_parser.extract_reasoning_with_tokens(text, token_ids=ids)
    parsed = tool_parser.extract_tool_calls_with_tokens(text, token_ids=ids)
    calls = [
        (call["name"], json.loads(call["arguments"])) for call in parsed.tool_calls
    ]
    return reasoning or "", parsed.content or "", calls


def _stream(tokenizer, ids, split):
    reasoning_parser = get_parser("inkling")(tokenizer)
    tool_parser = ToolParserManager.get_tool_parser("inkling")(tokenizer)
    decoder = IncrementalDecoder(tokenizer, skip_special_tokens=False)
    previous = tool_text = reasoning = content = ""
    pending_ids = []
    calls = []

    def consume(delta, delta_ids):
        nonlocal previous, tool_text, reasoning, content
        current = previous + delta
        msg = extract_reasoning_streaming(
            reasoning_parser, previous, current, delta, delta_token_ids=delta_ids
        )
        previous = current
        if not msg:
            return
        reasoning += msg.reasoning or ""
        text_delta = msg.content or ""
        if not text_delta:
            return
        before = tool_text
        tool_text += text_delta
        result = extract_tool_calls_streaming(
            tool_parser,
            before,
            tool_text,
            text_delta,
            delta_token_ids=msg.content_token_ids,
        )
        if result:
            content += result.get("content", "")
            calls.extend(result.get("tool_calls", []))

    for chunk in (ids[:split], ids[split:]):
        delta = "".join(decoder.add_token(token_id) for token_id in chunk)
        pending_ids.extend(chunk)
        if delta:
            consume(delta, tuple(pending_ids))
            pending_ids.clear()
    tail = reasoning_parser.finish_stream()
    if tail:
        reasoning += tail.reasoning or ""
        if tail.content:
            before = tool_text
            tool_text += tail.content
            result = extract_tool_calls_streaming(
                tool_parser,
                before,
                tool_text,
                tail.content,
                delta_token_ids=tail.content_token_ids,
            )
            if result:
                content += result.get("content", "")
                calls.extend(result.get("tool_calls", []))
    result = tool_parser.finalize_legacy_raw_stream(tool_text)
    if result:
        content += result.get("content", "")
        calls.extend(result.get("tool_calls", []))
    parsed_calls = [
        (call["function"]["name"], json.loads(call["function"]["arguments"]))
        for call in calls
    ]
    return reasoning, content, parsed_calls


def test_token_ids_recover_malformed_tool_and_later_blocks_complete_and_streaming():
    tokenizer, ids, text = _scenario()
    expected = (
        "Think before.Think after.",
        BAD + "Visible answer.Done é.",
        [("lookup", {"query": "ordinary query"})],
    )
    assert _complete(tokenizer, ids, text) == expected
    failures = [
        split
        for split in range(len(ids) + 1)
        if _stream(tokenizer, ids, split) != expected
    ]
    assert not failures, f"{len(failures)} token split points fail: {failures[:5]}"


def test_ordinary_ids_spelling_markers_remain_json_payload():
    literal = f"literal {E} and {R}"
    tokenizer, ids, text = _scenario(literal)
    expected = (
        "Think before.Think after.",
        BAD + "Visible answer.Done é.",
        [("lookup", {"query": literal})],
    )
    assert _complete(tokenizer, ids, text) == expected
    failures = [
        split
        for split in range(len(ids) + 1)
        if _stream(tokenizer, ids, split) != expected
    ]
    assert not failures, f"{len(failures)} token split points fail: {failures[:5]}"


def test_mismatched_synthetic_delta_falls_back_without_releasing_held_ids():
    tokenizer = Tokenizer()
    parser = get_parser("inkling")(tokenizer)
    held = [0xC3]
    msg = parser.extract_reasoning_streaming_with_tokens(
        "", "synthetic visible", "synthetic visible", delta_token_ids=held
    )
    assert msg.content == "synthetic visible"
    assert msg.content_token_ids is None
    assert parser.finish_stream() is None
    parser.reset_state()
    assert parser.extract_reasoning_with_tokens("plain", token_ids=[ord("p")]) == (
        None,
        "plain",
    )


def test_mixed_legacy_prefix_disables_token_mode_until_reset():
    tokenizer = Tokenizer()
    parser = get_parser("inkling")(tokenizer)
    assert parser.extract_reasoning_streaming("", "<|", "<|") is None
    msg = parser.extract_reasoning_streaming_with_tokens(
        "<|", "<|hello", "hello", delta_token_ids=list(b"hello")
    )
    assert msg.content == "<|hello"
    assert msg.content_token_ids is None
    parser.reset_state()
    msg = parser.extract_reasoning_streaming_with_tokens(
        "", "hello", "hello", delta_token_ids=list(b"hello")
    )
    assert msg.content == "hello"
    assert msg.content_token_ids == tuple(b"hello")


def test_malformed_final_tool_block_is_not_emitted_as_a_call():
    tokenizer, ids, text = _encode([M, C, BAD])
    tool = ToolParserManager.get_tool_parser("inkling")(tokenizer)
    result = tool.extract_tool_calls_with_tokens(text, token_ids=ids)
    assert not result.tools_called
    assert result.content == BAD

    streamed = ToolParserManager.get_tool_parser("inkling")(tokenizer)
    out = streamed.extract_tool_calls_streaming_with_tokens(
        "", text, text, delta_token_ids=ids
    )
    final = streamed.finalize_legacy_raw_stream(text)
    assert not (out or {}).get("tool_calls")
    assert not (final or {}).get("tool_calls")
    assert (out or {}).get("content", "") + (final or {}).get("content", "") == BAD


def test_postprocessor_forwards_token_ids_between_reasoning_and_tool_parsers():
    tokenizer, ids, _ = _scenario()
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = get_parser("inkling")(tokenizer)
    cfg.reasoning_parser_name = None
    cfg.enable_auto_tool_choice = False
    cfg.tool_call_parser = None
    cfg.tool_parser_instance = ToolParserManager.get_tool_parser("inkling")(tokenizer)
    pp = StreamingPostProcessor(
        cfg,
        tools_requested=True,
        enable_thinking=True,
        request={"tools": [{"type": "function", "function": {"name": "lookup"}}]},
    )
    decoder = IncrementalDecoder(tokenizer, skip_special_tokens=False)
    events = []
    for index, token_id in enumerate(ids):
        delta = decoder.add_token(token_id)
        events.extend(
            pp.process_chunk(
                GenerationOutput(
                    text=decoder.get_full_text(),
                    new_text=delta,
                    tokens=[token_id],
                    finished=index == len(ids) - 1,
                    finish_reason="stop" if index == len(ids) - 1 else None,
                )
            )
        )
    events.extend(pp.finalize())
    assert (
        "".join(event.reasoning or "" for event in events)
        == "Think before.Think after."
    )
    assert BAD + "Visible answer.Done é." in "".join(
        event.content or "" for event in events
    )
    assert any(event.type == "tool_call" for event in events)
