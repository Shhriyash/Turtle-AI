"""
test/telegram_webhook_test.py
--------------------------------
Unit coverage for apps/channels/telegram_webhook.py (Vercel migration
Phase 4) — the serverless-shaped replacement for the long-polling Telegram
gateway. Standalone FastAPI app with just this router mounted, mirroring
test/cron_tick_routes_test.py's pattern.
"""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import apps.channels.telegram_webhook as tw
from apps.channels import TurtleResponse
from apps.channels.telegram_webhook import router


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class SecretTokenAuthTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _make_client()

    def test_no_secret_configured_dev_anon_off_rejects(self) -> None:
        with patch.object(tw, "settings") as fake_settings:
            fake_settings.telegram_webhook_secret = None
            fake_settings.dev_anon = False
            fake_settings.is_cloud = False
            resp = self.client.post("/channels/telegram/webhook", json={})
        self.assertEqual(resp.status_code, 401)

    def test_no_secret_configured_dev_anon_on_local_allows(self) -> None:
        with patch.object(tw, "settings") as fake_settings:
            fake_settings.telegram_webhook_secret = None
            fake_settings.dev_anon = True
            fake_settings.is_cloud = False
            # No "message" key -> handled as a non-message update, 200 ack.
            resp = self.client.post("/channels/telegram/webhook", json={})
        self.assertEqual(resp.status_code, 200)

    def test_no_secret_configured_in_cloud_mode_rejects_even_with_dev_anon(self) -> None:
        with patch.object(tw, "settings") as fake_settings:
            fake_settings.telegram_webhook_secret = None
            fake_settings.dev_anon = True
            fake_settings.is_cloud = True
            resp = self.client.post("/channels/telegram/webhook", json={})
        self.assertEqual(resp.status_code, 401)

    def test_wrong_secret_token_rejected(self) -> None:
        with patch.object(tw, "settings") as fake_settings:
            fake_settings.telegram_webhook_secret.get_secret_value.return_value = "real-secret"
            resp = self.client.post(
                "/channels/telegram/webhook",
                json={},
                headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
            )
        self.assertEqual(resp.status_code, 401)

    def test_correct_secret_token_accepted(self) -> None:
        with patch.object(tw, "settings") as fake_settings:
            fake_settings.telegram_webhook_secret.get_secret_value.return_value = "real-secret"
            resp = self.client.post(
                "/channels/telegram/webhook",
                json={},
                headers={"X-Telegram-Bot-Api-Secret-Token": "real-secret"},
            )
        self.assertEqual(resp.status_code, 200)


class MessageHandlingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _make_client()
        self._settings_patcher = patch.object(tw, "settings")
        fake_settings = self._settings_patcher.start()
        fake_settings.telegram_webhook_secret.get_secret_value.return_value = "real-secret"
        self.addCleanup(self._settings_patcher.stop)
        self.headers = {"X-Telegram-Bot-Api-Secret-Token": "real-secret"}

    def _dm_payload(self, text: str, user_id: int = 111) -> dict:
        return {
            "message": {
                "message_id": 1,
                "text": text,
                "chat": {"id": 222, "type": "private"},
                "from": {"id": user_id, "is_bot": False, "first_name": "Alice"},
            }
        }

    def test_dm_dispatches_and_replies(self) -> None:
        with patch.object(
            tw, "_get_bot_username", new=AsyncMock(return_value="turtlebot")
        ), patch.object(
            tw, "resolve_channel_user", new_callable=AsyncMock, return_value="usr_a"
        ), patch.object(
            tw, "dispatch_event", new_callable=AsyncMock
        ) as fake_dispatch, patch.object(
            tw, "_send_reply", new_callable=AsyncMock
        ) as fake_send:
            fake_dispatch.return_value = TurtleResponse(
                content="hello back", channel="telegram", user_id="usr_a"
            )

            resp = self.client.post(
                "/channels/telegram/webhook", json=self._dm_payload("hi turtle"), headers=self.headers
            )

        self.assertEqual(resp.status_code, 200)
        fake_dispatch.assert_awaited_once()
        event = fake_dispatch.call_args[0][0]
        self.assertEqual(event.channel, "telegram")
        self.assertEqual(event.content, "hi turtle")
        self.assertTrue(event.is_private)
        fake_send.assert_awaited_once_with(222, "hello back")

    def test_bot_author_is_ignored(self) -> None:
        payload = self._dm_payload("hi")
        payload["message"]["from"]["is_bot"] = True
        with patch.object(tw, "dispatch_event", new_callable=AsyncMock) as fake_dispatch:
            resp = self.client.post(
                "/channels/telegram/webhook", json=payload, headers=self.headers
            )
        self.assertEqual(resp.status_code, 200)
        fake_dispatch.assert_not_called()

    def test_group_message_without_mention_is_ignored(self) -> None:
        payload = {
            "message": {
                "message_id": 1,
                "text": "just chatting",
                "chat": {"id": 333, "type": "group"},
                "from": {"id": 111, "is_bot": False},
            }
        }
        with patch.object(
            tw, "_get_bot_username", new=AsyncMock(return_value="turtlebot")
        ), patch.object(tw, "dispatch_event", new_callable=AsyncMock) as fake_dispatch:
            resp = self.client.post(
                "/channels/telegram/webhook", json=payload, headers=self.headers
            )
        self.assertEqual(resp.status_code, 200)
        fake_dispatch.assert_not_called()

    def test_group_message_with_mention_is_handled(self) -> None:
        payload = {
            "message": {
                "message_id": 1,
                "text": "@turtlebot what's up",
                "chat": {"id": 333, "type": "group"},
                "from": {"id": 111, "is_bot": False},
            }
        }
        with patch.object(
            tw, "_get_bot_username", new=AsyncMock(return_value="turtlebot")
        ), patch.object(
            tw, "resolve_channel_user", new_callable=AsyncMock, return_value="usr_a"
        ), patch.object(
            tw, "dispatch_event", new_callable=AsyncMock
        ) as fake_dispatch, patch.object(
            tw, "_send_reply", new_callable=AsyncMock
        ):
            fake_dispatch.return_value = TurtleResponse(
                content="hi", channel="telegram", user_id="usr_a"
            )
            resp = self.client.post(
                "/channels/telegram/webhook", json=payload, headers=self.headers
            )
        self.assertEqual(resp.status_code, 200)
        fake_dispatch.assert_awaited_once()
        event = fake_dispatch.call_args[0][0]
        self.assertEqual(event.content, "what's up")  # mention stripped
        self.assertFalse(event.is_private)

    def test_empty_text_after_mention_strip_is_ignored(self) -> None:
        payload = self._dm_payload("")
        with patch.object(
            tw, "_get_bot_username", new=AsyncMock(return_value="turtlebot")
        ), patch.object(tw, "dispatch_event", new_callable=AsyncMock) as fake_dispatch:
            resp = self.client.post(
                "/channels/telegram/webhook", json=payload, headers=self.headers
            )
        self.assertEqual(resp.status_code, 200)
        fake_dispatch.assert_not_called()

    def test_non_message_update_is_acked_and_ignored(self) -> None:
        with patch.object(tw, "dispatch_event", new_callable=AsyncMock) as fake_dispatch:
            resp = self.client.post(
                "/channels/telegram/webhook",
                json={"channel_post": {"text": "not a dm"}},
                headers=self.headers,
            )
        self.assertEqual(resp.status_code, 200)
        fake_dispatch.assert_not_called()

    def test_dispatch_failure_sends_error_reply_not_500(self) -> None:
        with patch.object(
            tw, "_get_bot_username", new=AsyncMock(return_value="turtlebot")
        ), patch.object(
            tw, "resolve_channel_user", new_callable=AsyncMock, return_value="usr_a"
        ), patch.object(
            tw, "dispatch_event", new_callable=AsyncMock, side_effect=RuntimeError("boom")
        ), patch.object(
            tw, "_send_reply", new_callable=AsyncMock
        ) as fake_send:
            resp = self.client.post(
                "/channels/telegram/webhook", json=self._dm_payload("hi"), headers=self.headers
            )
        self.assertEqual(resp.status_code, 200)
        fake_send.assert_awaited_once()
        self.assertIn("wrong", fake_send.call_args[0][1].lower())

    def test_invite_only_unknown_sender_no_mint_no_dispatch(self) -> None:
        """WP 1.D (ledger 1a.4): TURTLE_CHANNEL_SIGNUP=invite + unknown
        sender -> resolve_channel_user returns None (never mints), the
        adapter replies with the invite message, and no turn is dispatched.
        """
        with patch.object(
            tw, "_get_bot_username", new=AsyncMock(return_value="turtlebot")
        ), patch.object(
            tw, "resolve_channel_user", new_callable=AsyncMock, return_value=None
        ) as fake_resolve, patch.object(
            tw, "dispatch_event", new_callable=AsyncMock
        ) as fake_dispatch, patch.object(
            tw, "_send_reply", new_callable=AsyncMock
        ) as fake_send:
            resp = self.client.post(
                "/channels/telegram/webhook", json=self._dm_payload("hi"), headers=self.headers
            )
        self.assertEqual(resp.status_code, 200)
        fake_resolve.assert_awaited_once_with("telegram", "111")
        fake_dispatch.assert_not_called()
        fake_send.assert_awaited_once_with(222, tw.CHANNEL_INVITE_ONLY_MESSAGE)


class PureHelperTest(unittest.TestCase):
    def test_extract_message_text_strips_leading_mention(self) -> None:
        text = tw._extract_message_text({"text": "@turtlebot hello there"}, "turtlebot")
        self.assertEqual(text, "hello there")

    def test_extract_message_text_no_mention_passthrough(self) -> None:
        text = tw._extract_message_text({"text": "hello there"}, "turtlebot")
        self.assertEqual(text, "hello there")

    def test_should_handle_dm_always_true_unless_bot(self) -> None:
        self.assertTrue(tw._should_handle("private", is_mention=False, is_bot_author=False))
        self.assertFalse(tw._should_handle("private", is_mention=False, is_bot_author=True))

    def test_should_handle_group_requires_mention(self) -> None:
        self.assertFalse(tw._should_handle("group", is_mention=False, is_bot_author=False))
        self.assertTrue(tw._should_handle("group", is_mention=True, is_bot_author=False))

    def test_verify_secret_token_constant_time_compare(self) -> None:
        with patch.object(tw, "settings") as fake_settings:
            fake_settings.telegram_webhook_secret.get_secret_value.return_value = "abc123"
            self.assertTrue(tw._verify_secret_token("abc123"))
            self.assertFalse(tw._verify_secret_token("abc124"))
            self.assertFalse(tw._verify_secret_token(""))


if __name__ == "__main__":
    unittest.main()
