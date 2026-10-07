from serana_agent.llm.toolcall import parse_output, strip_think


def test_strip_think_closed_block():
    assert strip_think("<think>\nplan\n</think>\n\nhello") == "hello"


def test_strip_think_only_closing_tag():
    # The template already opened <think>, so the output starts mid-thought.
    assert strip_think("plan...\n</think>\n\nhello") == "hello"


def test_strip_think_unclosed_drops_rest():
    assert strip_think("hi <think>still thinking") == "hi"


def test_unclosed_think_is_parse_error():
    content, calls, err = parse_output("<think>\nstill thinking about")
    assert (content, calls) == ("", []) and "<think>" in err


def test_empty_output_is_parse_error():
    assert parse_output("  ")[2] == "empty output"


def test_closing_think_inside_arguments_is_data():
    text = (
        "<think>\nplan\n</think>\n"
        '<tool_call>{"name": "write_file", "arguments": {"path": "a", "content": "x</think>y"}}'
        "</tool_call>"
    )
    _, calls, err = parse_output(text)
    assert err is None and calls[0].arguments["content"] == "x</think>y"
    # no leading think at all, closing tag only inside the arguments
    _, calls, err = parse_output(text.split("\n", 3)[3])
    assert err is None and calls[0].arguments["content"] == "x</think>y"


def test_raw_newline_in_json_string():
    text = (
        '<tool_call>{"name": "write_file", "arguments": {"path": "a", '
        '"content": "l1\nl2"}}</tool_call>'
    )
    _, calls, err = parse_output(text)
    assert err is None and calls[0].arguments["content"] == "l1\nl2"


def test_plain_text():
    content, calls, err = parse_output("그냥 대답이야.")
    assert (content, calls, err) == ("그냥 대답이야.", [], None)


def test_single_tool_call():
    text = (
        "<think>\n\n</think>\n\n<tool_call>\n"
        '{"name": "read_file", "arguments": {"path": "a.txt"}}\n</tool_call>'
    )
    content, calls, err = parse_output(text)
    assert content == "" and err is None
    assert [(c.name, c.arguments) for c in calls] == [("read_file", {"path": "a.txt"})]
    assert calls[0].id


def test_text_before_and_multiple_calls():
    text = (
        "먼저 읽을게.\n"
        '<tool_call>\n{"name": "list_dir", "arguments": {"path": "."}}\n</tool_call>\n'
        '<tool_call>\n{"name": "list_notes", "arguments": {}}\n</tool_call>'
    )
    content, calls, err = parse_output(text)
    assert content == "먼저 읽을게." and err is None
    assert [c.name for c in calls] == ["list_dir", "list_notes"]
    assert calls[0].id != calls[1].id


def test_arguments_as_json_string():
    text = '<tool_call>{"name": "read_file", "arguments": "{\\"path\\": \\"a\\"}"}</tool_call>'
    _, calls, err = parse_output(text)
    assert err is None and calls[0].arguments == {"path": "a"}


def test_missing_arguments_defaults_to_empty():
    _, calls, err = parse_output('<tool_call>{"name": "list_notes"}</tool_call>')
    assert err is None and calls[0].arguments == {}


def test_broken_json_sets_parse_error():
    content, calls, err = parse_output(
        '<tool_call>\n{"name": "read_file", "arguments": {\n</tool_call>'
    )
    assert calls == [] and err and "invalid JSON" in err


def test_unclosed_block_sets_parse_error():
    content, calls, err = parse_output('ok <tool_call>\n{"name": "read_file", "argu')
    assert calls == [] and content == "ok" and err and "not closed" in err


def test_missing_name_and_bad_arguments():
    _, calls, err = parse_output('<tool_call>{"arguments": {}}</tool_call>')
    assert calls == [] and "name" in err
    _, calls, err = parse_output('<tool_call>{"name": "x", "arguments": [1]}</tool_call>')
    assert calls == [] and "JSON object" in err


def test_valid_call_kept_when_another_is_broken():
    text = (
        '<tool_call>{"name": "list_notes", "arguments": {}}</tool_call>'
        "<tool_call>{oops}</tool_call>"
    )
    _, calls, err = parse_output(text)
    assert len(calls) == 1 and err


def test_tool_call_inside_think_is_ignored():
    text = '<think>maybe <tool_call>{"name": "x"}</tool_call></think>done'
    content, calls, err = parse_output(text)
    assert (content, calls, err) == ("done", [], None)
