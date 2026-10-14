"""Clarify choices follow ``extra.clarify_choices_in_text``: full text in the message, numbered buttons."""

from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_slack_clarify_buttons import _attach_auth_runner, _clear_clarify_state, _make_adapter

SECTION_TEXT_LIMIT = 3000
BUTTON_TEXT_LIMIT = 75
QUESTION = "How should the Newsletter table pick up Email Data?"
LONG_CHOICES = [
    "Create one Newsletter row per Email Data row, carrying its send time, audience, delivered, opened and clicked counts",
    "Sum each month's Paid and Unpaid Delivered counts and fill them into the Newsletter row holding that month's slot",
    "Leave the 2023 placeholder rows alone and only add rows for months that have no Newsletter row yet at all",
]


def _adapter(listed):
    _clear_clarify_state()
    adapter = _make_adapter()
    if listed is not None:
        adapter.config.extra["clarify_choices_in_text"] = listed
    client = adapter._team_clients["T1"]
    client.chat_postMessage = AsyncMock(return_value={"ts": "9.9"})
    client.chat_update = AsyncMock()
    return adapter, client


async def _posted_blocks(adapter, client, choices, clarify_id="cid", question=QUESTION):
    result = await adapter.send_clarify(
        chat_id="C1", question=question, choices=choices, clarify_id=clarify_id, session_key="sk")
    assert result.success
    return client.chat_postMessage.call_args.kwargs["blocks"]


def _buttons(blocks):
    return [element for block in blocks if block["type"] == "actions" for element in block["elements"]]


def _choice_buttons(blocks):
    return [button for button in _buttons(blocks) if button["action_id"].startswith("hermes_clarify_choice_")]


def _choice_sections(blocks):
    return [block["text"]["text"] for block in blocks if block["type"] == "section"][1:]


@pytest.mark.asyncio
@pytest.mark.parametrize("listed", [None, False, "false"])
async def test_switch_off_keeps_each_choice_on_its_own_button(listed):
    adapter, client = _adapter(listed)
    blocks = await _posted_blocks(adapter, client, LONG_CHOICES)
    assert not _choice_sections(blocks)
    labels = [button["text"]["text"] for button in _choice_buttons(blocks)]
    assert labels == [choice[:BUTTON_TEXT_LIMIT] for choice in LONG_CHOICES]


@pytest.mark.asyncio
@pytest.mark.parametrize("listed", [True, "true"])
async def test_every_choice_shows_in_full_under_the_number_its_button_shows(listed):
    adapter, client = _adapter(listed)
    blocks = await _posted_blocks(adapter, client, LONG_CHOICES)
    sections = _choice_sections(blocks)
    buttons = _choice_buttons(blocks)
    assert QUESTION in blocks[0]["text"]["text"]
    assert len(sections) == len(buttons) == len(LONG_CHOICES)
    for idx, (choice, section, button) in enumerate(zip(LONG_CHOICES, sections, buttons)):
        number = button["text"]["text"]
        assert len(number) < len(choice) and number in section.split(choice)[0]
        assert choice in section
        assert button["value"] == f"cid|{idx}"
    assert len({button["text"]["text"] for button in buttons}) == len(buttons)
    assert _buttons(blocks)[-1]["action_id"] == "hermes_clarify_other"


@pytest.mark.asyncio
async def test_a_numbered_button_resolves_to_its_choice_and_the_card_keeps_question_and_full_answer():
    from tools import clarify_gateway as cm

    adapter, client = _adapter(True)
    _attach_auth_runner(adapter)
    cm.register("cid", "sk", QUESTION, LONG_CHOICES)
    blocks = await _posted_blocks(adapter, client, LONG_CHOICES)
    button = _choice_buttons(blocks)[1]

    await adapter._handle_clarify_action(AsyncMock(), {
        "message": {"ts": "9.9", "blocks": blocks},
        "channel": {"id": "C1"}, "user": {"name": "ryan", "id": "U_R"},
    }, {"action_id": button["action_id"], "value": button["value"]})

    assert cm._entries["cid"].response == LONG_CHOICES[1]
    card = client.chat_update.call_args.kwargs["blocks"]
    assert QUESTION in card[0]["text"]["text"]
    assert all(choice not in card[0]["text"]["text"] for choice in LONG_CHOICES)
    assert LONG_CHOICES[1] in card[-1]["elements"][0]["text"]
    assert not _buttons(card)


@pytest.mark.asyncio
async def test_a_retired_card_keeps_only_the_question():
    adapter, client = _adapter(True)
    await _posted_blocks(adapter, client, LONG_CHOICES)
    await adapter.retire_clarify_card("cid", "expired")
    card = client.chat_update.call_args.kwargs["blocks"]
    texts = [block["text"]["text"] for block in card if block["type"] == "section"]
    assert texts == [card[0]["text"]["text"]] and QUESTION in texts[0]
    assert all(choice not in texts[0] for choice in LONG_CHOICES) and not _buttons(card)


@pytest.mark.asyncio
async def test_mentions_and_links_in_a_choice_show_as_text():
    hostile = ["<!channel> ping <@U0ADMIN> & <https://evil.example|open the report>", "plain"]
    adapter, client = _adapter(True)
    blocks = await _posted_blocks(adapter, client, hostile)
    [first, _second] = _choice_sections(blocks)
    assert "<" not in first and ">" not in first
    assert "&lt;!channel&gt;" in first and "&lt;@U0ADMIN&gt;" in first and " &amp; " in first


@pytest.mark.asyncio
async def test_choices_too_long_for_one_section_still_line_up_with_their_buttons():
    huge = [f"{idx}" * (SECTION_TEXT_LIMIT + 500) for idx in range(4)]
    adapter, client = _adapter(True)
    blocks = await _posted_blocks(adapter, client, huge, question="Q" * (SECTION_TEXT_LIMIT + 500))
    sections = _choice_sections(blocks)
    buttons = _choice_buttons(blocks)
    assert all(len(block["text"]["text"]) <= SECTION_TEXT_LIMIT for block in blocks if block["type"] == "section")
    assert len(sections) == len(buttons) == len(huge)
    for choice, section, button in zip(huge, sections, buttons):
        assert section.startswith(f"*{button['text']['text']}.*") and choice[:100] in section
    assert all(len(block["elements"]) <= 5 for block in blocks if block["type"] == "actions")


@pytest.mark.asyncio
async def test_a_blank_choice_still_gets_its_number():
    adapter, client = _adapter(True)
    blocks = await _posted_blocks(adapter, client, ["  ", "second"])
    sections = _choice_sections(blocks)
    buttons = _choice_buttons(blocks)
    assert len(sections) == len(buttons) == 2
    assert [button["value"] for button in buttons] == ["cid|0", "cid|1"]
    assert all(section.startswith(f"*{button['text']['text']}.*") for section, button in zip(sections, buttons))
