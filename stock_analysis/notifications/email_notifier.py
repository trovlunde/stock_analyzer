import logging
import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SmtpConfig:
    host: str
    port: int
    username: str | None
    password: str | None
    use_tls: bool
    sender: str
    recipients: list[str]


def load_smtp_config_from_env() -> SmtpConfig | None:
    """Load SMTP settings from the environment. Returns None if alerting is unconfigured —
    unlike DATABASE_URL (which has a working default), unset SMTP_HOST means alerting is off."""
    load_dotenv()
    host = os.environ.get("SMTP_HOST")
    if not host:
        return None

    recipients = [r.strip() for r in os.environ.get("SMTP_TO", "").split(",") if r.strip()]
    username = os.environ.get("SMTP_USER") or None
    raw_port = os.environ.get("SMTP_PORT", "587")
    try:
        port = int(raw_port)
    except ValueError:
        logger.warning("Invalid SMTP_PORT %r — alerting disabled", raw_port)
        return None

    if not recipients:
        logger.warning("SMTP_HOST is set but SMTP_TO has no recipients — alerting disabled")
        return None

    return SmtpConfig(
        host=host,
        port=port,
        username=username,
        password=os.environ.get("SMTP_PASSWORD") or None,
        use_tls=os.environ.get("SMTP_USE_TLS", "true").lower() == "true",
        sender=os.environ.get("SMTP_FROM") or username or host,
        recipients=recipients,
    )


class EmailNotifier:
    def __init__(self, config: SmtpConfig | None):
        self.config = config

    def send_daily_summary(self, report_path: Path, summary: dict) -> None:
        lines = [f"{key}: {value}" for key, value in summary.items()]
        body = "\n".join(lines) + f"\n\nFull report: {report_path}"
        self._send("Daily trading run summary", body)

    def send_error_alert(self, subject: str, error: BaseException | str) -> None:
        self._send(f"ALERT: {subject}", str(error))

    def _send(self, subject: str, body: str) -> None:
        if self.config is None or not self.config.recipients:
            logger.info("SMTP not configured — skipping alert: %s", subject)
            return

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.config.sender
        message["To"] = ", ".join(self.config.recipients)
        message.set_content(body)

        try:
            with smtplib.SMTP(self.config.host, self.config.port, timeout=30) as client:
                if self.config.use_tls:
                    client.starttls()
                if self.config.username:
                    client.login(self.config.username, self.config.password or "")
                client.send_message(message)
        except Exception:
            logger.exception("Failed to send email alert: %s", subject)
