#!/usr/bin/env python3
"""
EIMAS Notification System
=========================
Send alerts via Slack and Email when signals are generated.

Environment Variables:
    SLACK_WEBHOOK_URL: Slack webhook URL for sending messages
    SMTP_SERVER: SMTP server for email (default: smtp.gmail.com)
    SMTP_PORT: SMTP port (default: 587)
    SMTP_USER: Email username
    SMTP_PASSWORD: Email app password
    ALERT_EMAIL: Email to send alerts to

Usage:
    from lib.notifications import NotificationService

    notifier = NotificationService()
    notifier.send_signal_alert(signals, consensus)
"""

import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import httpx


@dataclass
class NotificationConfig:
    """Configuration for notifications"""
    # Slack
    slack_webhook_url: Optional[str] = None

    # Email
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    alert_email: Optional[str] = None

    # Settings
    min_conviction_alert: float = 0.6  # Only alert if conviction >= this
    alert_on_actions: List[str] = None  # Only alert on these actions

    def __post_init__(self):
        if self.alert_on_actions is None:
            self.alert_on_actions = ['buy', 'sell', 'reduce']


class SlackNotifier:
    """Send notifications to Slack"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, message: str, blocks: List[Dict] = None) -> bool:
        """Send message to Slack"""
        payload = {"text": message}
        if blocks:
            payload["blocks"] = blocks

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(self.webhook_url, json=payload)
                return response.status_code == 200
        except Exception as e:
            print(f"Slack error: {e}")
            return False

    def format_signal_alert(
        self,
        signals: List[Dict],
        consensus: Dict
    ) -> tuple[str, List[Dict]]:
        """Format signals as Slack blocks"""
        action = consensus.get('action', 'N/A').upper()
        conviction = consensus.get('conviction', 0)

        # Emoji mapping
        emoji_map = {
            'BUY': ':chart_with_upwards_trend:',
            'SELL': ':chart_with_downwards_trend:',
            'HOLD': ':pause_button:',
            'REDUCE': ':warning:',
            'HEDGE': ':shield:'
        }
        emoji = emoji_map.get(action, ':bell:')

        # Plain text fallback
        text = f"EIMAS Alert: {action} signal ({conviction:.0%} conviction)"

        # Rich blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} EIMAS Market Alert",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Recommendation:*\n{action}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Conviction:*\n{conviction:.0%}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Total Signals:*\n{len(signals)}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    }
                ]
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Signal Breakdown:*"
                }
            }
        ]

        # Add each signal
        for s in signals[:5]:  # Limit to 5 signals
            signal_emoji = emoji_map.get(s.get('action', '').upper(), '')
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{signal_emoji} *{s.get('source', 'N/A')}*: {s.get('action', 'N/A').upper()} ({s.get('conviction', 0):.0%})\n_{s.get('reasoning', '')[:100]}_"
                }
            })

        # Add reasoning
        reasoning = consensus.get('reasoning', '')
        if reasoning:
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Key Reasoning:*\n{reasoning[:500]}"
                }
            })

        return text, blocks


class EmailNotifier:
    """Send notifications via Email"""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        smtp_user: str,
        smtp_password: str
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password

    def send(
        self,
        to_email: str,
        subject: str,
        body_html: str,
        body_text: str = None
    ) -> bool:
        """Send email"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.smtp_user
            msg['To'] = to_email

            # Plain text version
            if body_text:
                msg.attach(MIMEText(body_text, 'plain'))

            # HTML version
            msg.attach(MIMEText(body_html, 'html'))

            # Send
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            return True
        except Exception as e:
            print(f"Email error: {e}")
            return False

    def format_signal_alert(
        self,
        signals: List[Dict],
        consensus: Dict
    ) -> tuple[str, str, str]:
        """Format signals as email HTML"""
        action = consensus.get('action', 'N/A').upper()
        conviction = consensus.get('conviction', 0)
        reasoning = consensus.get('reasoning', '')

        subject = f"EIMAS Alert: {action} ({conviction:.0%} conviction)"

        # Plain text
        text = f"""
EIMAS Market Alert
==================
Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}

RECOMMENDATION: {action}
CONVICTION: {conviction:.0%}
SIGNALS: {len(signals)}

Signal Breakdown:
"""
        for s in signals:
            text += f"- {s.get('source', 'N/A')}: {s.get('action', 'N/A').upper()} ({s.get('conviction', 0):.0%})\n"
            text += f"  {s.get('reasoning', '')}\n\n"

        text += f"\nKey Reasoning:\n{reasoning}"

        # HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #1f77b4; color: white; padding: 20px; border-radius: 5px; }}
        .metrics {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric {{ background: #f0f2f6; padding: 15px; border-radius: 5px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .signal-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .signal-table th, .signal-table td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        .signal-table th {{ background: #f0f2f6; }}
        .buy {{ color: #28a745; }}
        .sell {{ color: #dc3545; }}
        .reduce {{ color: #fd7e14; }}
        .hedge {{ color: #17a2b8; }}
        .hold {{ color: #6c757d; }}
        .reasoning {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 EIMAS Market Alert</h1>
        <p>{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>

    <div class="metrics">
        <div class="metric">
            <div class="metric-value {action.lower()}">{action}</div>
            <div>Recommendation</div>
        </div>
        <div class="metric">
            <div class="metric-value">{conviction:.0%}</div>
            <div>Conviction</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(signals)}</div>
            <div>Signals</div>
        </div>
    </div>

    <h2>Signal Breakdown</h2>
    <table class="signal-table">
        <tr>
            <th>Source</th>
            <th>Action</th>
            <th>Conviction</th>
            <th>Reasoning</th>
        </tr>
"""
        for s in signals:
            action_class = s.get('action', 'hold').lower()
            html += f"""
        <tr>
            <td>{s.get('source', 'N/A')}</td>
            <td class="{action_class}">{s.get('action', 'N/A').upper()}</td>
            <td>{s.get('conviction', 0):.0%}</td>
            <td>{s.get('reasoning', '')[:100]}...</td>
        </tr>
"""

        html += f"""
    </table>

    <div class="reasoning">
        <h3>Key Reasoning</h3>
        <p>{reasoning}</p>
    </div>

    <hr>
    <p style="color: #666; font-size: 12px;">
        Generated by EIMAS (Economic Intelligence Multi-Agent System)
    </p>
</body>
</html>
"""

        return subject, html, text


class NotificationService:
    """Unified notification service"""

    def __init__(self, config: NotificationConfig = None):
        self.config = config or self._load_config_from_env()
        self.slack = None
        self.email = None

        # Initialize Slack
        if self.config.slack_webhook_url:
            self.slack = SlackNotifier(self.config.slack_webhook_url)
            print("  Slack notifications enabled")

        # Initialize Email
        if self.config.smtp_user and self.config.smtp_password:
            self.email = EmailNotifier(
                self.config.smtp_server,
                self.config.smtp_port,
                self.config.smtp_user,
                self.config.smtp_password
            )
            print("  Email notifications enabled")

    def _load_config_from_env(self) -> NotificationConfig:
        """Load configuration from environment variables"""
        return NotificationConfig(
            slack_webhook_url=os.environ.get('SLACK_WEBHOOK_URL'),
            smtp_server=os.environ.get('SMTP_SERVER', 'smtp.gmail.com'),
            smtp_port=int(os.environ.get('SMTP_PORT', '587')),
            smtp_user=os.environ.get('SMTP_USER'),
            smtp_password=os.environ.get('SMTP_PASSWORD'),
            alert_email=os.environ.get('ALERT_EMAIL'),
            min_conviction_alert=float(os.environ.get('MIN_CONVICTION_ALERT', '0.5'))
        )

    def should_alert(self, consensus: Dict) -> bool:
        """Determine if we should send an alert"""
        action = consensus.get('action', 'hold')
        conviction = consensus.get('conviction', 0)

        # Check conviction threshold
        if conviction < self.config.min_conviction_alert:
            return False

        # Check action filter
        if self.config.alert_on_actions and action not in self.config.alert_on_actions:
            return False

        return True

    def send_signal_alert(
        self,
        signals: List[Dict],
        consensus: Dict,
        force: bool = False
    ) -> Dict[str, bool]:
        """Send signal alert to all configured channels"""
        results = {'slack': False, 'email': False}

        # Check if we should alert
        if not force and not self.should_alert(consensus):
            print("  Alert skipped (below threshold)")
            return results

        print("  Sending notifications...")

        # Send to Slack
        if self.slack:
            text, blocks = self.slack.format_signal_alert(signals, consensus)
            results['slack'] = self.slack.send(text, blocks)
            print(f"    Slack: {'✓' if results['slack'] else '✗'}")

        # Send to Email
        if self.email and self.config.alert_email:
            subject, html, text = self.email.format_signal_alert(signals, consensus)
            results['email'] = self.email.send(
                self.config.alert_email,
                subject,
                html,
                text
            )
            print(f"    Email: {'✓' if results['email'] else '✗'}")

        return results

    def send_test_notification(self) -> Dict[str, bool]:
        """Send a test notification"""
        test_signals = [
            {
                'source': 'test',
                'action': 'buy',
                'conviction': 0.75,
                'reasoning': 'This is a test notification from EIMAS'
            }
        ]
        test_consensus = {
            'action': 'buy',
            'conviction': 0.75,
            'signal_count': 1,
            'reasoning': 'Test alert - system is working correctly'
        }

        return self.send_signal_alert(test_signals, test_consensus, force=True)


# ============================================================================
# Integration with Pipeline
# ============================================================================

def run_pipeline_with_notifications():
    """Run signal pipeline and send notifications"""
    from lib.signal_pipeline import SignalPipeline

    print("\n" + "=" * 60)
    print("EIMAS Pipeline with Notifications")
    print("=" * 60)

    # Run pipeline
    pipeline = SignalPipeline()
    signals = pipeline.run()
    consensus = pipeline.get_consensus()

    # Convert to dict format
    signal_dicts = []
    for s in signals:
        signal_dicts.append({
            'source': s.source.value,
            'action': s.action.value,
            'ticker': s.ticker,
            'conviction': s.conviction,
            'reasoning': s.reasoning
        })

    # Send notifications
    notifier = NotificationService()
    results = notifier.send_signal_alert(signal_dicts, consensus)

    return results


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EIMAS Notification System Test")
    print("=" * 60)

    # Check configuration
    config = NotificationConfig(
        slack_webhook_url=os.environ.get('SLACK_WEBHOOK_URL'),
        smtp_user=os.environ.get('SMTP_USER'),
        smtp_password=os.environ.get('SMTP_PASSWORD'),
        alert_email=os.environ.get('ALERT_EMAIL')
    )

    print(f"\nConfiguration:")
    print(f"  Slack Webhook: {'Configured' if config.slack_webhook_url else 'Not set'}")
    print(f"  SMTP User: {'Configured' if config.smtp_user else 'Not set'}")
    print(f"  Alert Email: {'Configured' if config.alert_email else 'Not set'}")

    if config.slack_webhook_url or config.smtp_user:
        print("\nSending test notification...")
        notifier = NotificationService(config)
        results = notifier.send_test_notification()
        print(f"\nResults: {results}")
    else:
        print("\nNo notification channels configured.")
        print("Set environment variables to enable:")
        print("  - SLACK_WEBHOOK_URL: for Slack notifications")
        print("  - SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL: for Email notifications")
