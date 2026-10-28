from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stock_analysis.notifications.email_notifier import (
    EmailNotifier,
    SmtpConfig,
    load_smtp_config_from_env,
)


@pytest.fixture
def smtp_env(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "2525")
    monkeypatch.setenv("SMTP_USER", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    monkeypatch.setenv("SMTP_FROM", "alerts@example.com")
    monkeypatch.setenv("SMTP_TO", "a@example.com, b@example.com")
    monkeypatch.setenv("SMTP_USE_TLS", "true")


def test_load_smtp_config_from_env_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    assert load_smtp_config_from_env() is None


def test_load_smtp_config_from_env_returns_none_on_invalid_port(smtp_env, monkeypatch):
    monkeypatch.setenv("SMTP_PORT", "not-a-port")
    assert load_smtp_config_from_env() is None


def test_load_smtp_config_from_env_returns_none_without_recipients(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.delenv("SMTP_TO", raising=False)
    assert load_smtp_config_from_env() is None


def test_load_smtp_config_from_env_parses_values(smtp_env):
    config = load_smtp_config_from_env()
    assert config == SmtpConfig(
        host="smtp.example.com",
        port=2525,
        username="bot@example.com",
        password="secret",
        use_tls=True,
        sender="alerts@example.com",
        recipients=["a@example.com", "b@example.com"],
    )


def test_notifier_skips_send_when_unconfigured(monkeypatch):
    mock_smtp_cls = MagicMock()
    monkeypatch.setattr("smtplib.SMTP", mock_smtp_cls)

    EmailNotifier(None).send_daily_summary(Path("report.md"), {"cash": 100})

    mock_smtp_cls.assert_not_called()


def test_notifier_sends_daily_summary(monkeypatch, smtp_env):
    mock_client = MagicMock()
    mock_smtp_cls = MagicMock()
    mock_smtp_cls.return_value.__enter__.return_value = mock_client
    monkeypatch.setattr("smtplib.SMTP", mock_smtp_cls)

    config = load_smtp_config_from_env()
    EmailNotifier(config).send_daily_summary(Path("report.md"), {"cash": 100})

    mock_smtp_cls.assert_called_once_with("smtp.example.com", 2525, timeout=30)
    mock_client.starttls.assert_called_once()
    mock_client.login.assert_called_once_with("bot@example.com", "secret")
    assert mock_client.send_message.call_count == 1
    sent = mock_client.send_message.call_args[0][0]
    assert sent["Subject"] == "Daily trading run summary"
    assert sent["To"] == "a@example.com, b@example.com"


def test_notifier_error_alert_subject_prefixed(monkeypatch, smtp_env):
    mock_client = MagicMock()
    mock_smtp_cls = MagicMock()
    mock_smtp_cls.return_value.__enter__.return_value = mock_client
    monkeypatch.setattr("smtplib.SMTP", mock_smtp_cls)

    config = load_smtp_config_from_env()
    EmailNotifier(config).send_error_alert("Daily job failed", RuntimeError("boom"))

    sent = mock_client.send_message.call_args[0][0]
    assert sent["Subject"] == "ALERT: Daily job failed"
    assert "boom" in sent.get_content()


def test_notifier_swallows_smtp_errors(monkeypatch, smtp_env):
    mock_smtp_cls = MagicMock(side_effect=OSError("connection refused"))
    monkeypatch.setattr("smtplib.SMTP", mock_smtp_cls)

    config = load_smtp_config_from_env()
    # Must not raise.
    EmailNotifier(config).send_daily_summary(Path("report.md"), {"cash": 100})
