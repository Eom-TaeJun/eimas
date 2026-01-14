#!/usr/bin/env python3
"""
EIMAS Notification System
=========================
Telegram/Slack/Discord 알림 발송

사용법:
    1. 환경변수 설정:
       export TELEGRAM_BOT_TOKEN="your_bot_token"
       export TELEGRAM_CHAT_ID="your_chat_id"

    2. 봇 토큰 받기: @BotFather에서 /newbot
    3. 채팅 ID 받기: @userinfobot에서 확인

    from lib.notifier import TelegramNotifier
    notifier = TelegramNotifier()
    notifier.send("Hello!")
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import os
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json


# ============================================================================
# Constants
# ============================================================================

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
SLACK_WEBHOOK_URL = os.environ.get('SLACK_WEBHOOK_URL', '')


class AlertLevel(str, Enum):
    """알림 레벨"""
    INFO = "info"           # 일반 정보
    WARNING = "warning"     # 주의
    CRITICAL = "critical"   # 긴급


# ============================================================================
# Telegram Notifier
# ============================================================================

class TelegramNotifier:
    """
    Telegram 알림 봇

    사용법:
        notifier = TelegramNotifier()

        # 단순 메시지
        notifier.send("시장 분석 완료!")

        # 포맷된 메시지
        notifier.send_alert(
            title="VIX 급등",
            message="VIX가 30을 돌파했습니다.",
            level=AlertLevel.CRITICAL
        )

        # 시장 요약
        notifier.send_market_summary(summary_dict)
    """

    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"

    def is_configured(self) -> bool:
        """설정 확인"""
        return bool(self.token and self.chat_id)

    def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        메시지 발송

        Args:
            message: 발송할 메시지 (HTML 포맷 지원)
            parse_mode: HTML 또는 Markdown

        Returns:
            성공 여부
        """
        if not self.is_configured():
            print("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
            return False

        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': parse_mode,
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Telegram send failed: {e}")
            return False

    def send_alert(self, title: str, message: str,
                   level: AlertLevel = AlertLevel.INFO) -> bool:
        """
        포맷된 알림 발송

        Args:
            title: 알림 제목
            message: 알림 내용
            level: 알림 레벨

        Returns:
            성공 여부
        """
        # 레벨별 아이콘
        icons = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨",
        }
        icon = icons.get(level, "📢")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        formatted = f"""
{icon} <b>{title}</b>

{message}

<i>{timestamp}</i>
"""
        return self.send(formatted.strip())

    def send_market_summary(self, data: Dict[str, Any]) -> bool:
        """
        시장 요약 발송

        Args:
            data: 시장 데이터 딕셔너리

        Returns:
            성공 여부
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 기본 정보 추출
        spy = data.get('spy_close', 0)
        spy_change = data.get('spy_change', 0)
        vix = data.get('vix', 0)
        btc = data.get('btc_price', 0)
        fear_greed = data.get('fear_greed_value', 50)
        fear_label = data.get('fear_greed_label', 'Neutral')

        message = f"""
📊 <b>EIMAS 시장 요약</b>

<b>주요 지표</b>
• SPY: ${spy:,.2f} ({spy_change:+.2f}%)
• VIX: {vix:.1f}
• BTC: ${btc:,.0f}
• Fear & Greed: {fear_greed} ({fear_label})
"""

        # 신호 추가
        signals = data.get('signals', [])
        if signals:
            message += "\n<b>📈 Signals</b>\n"
            for sig in signals[:5]:  # 최대 5개
                message += f"• {sig}\n"

        # 경고 추가
        warnings = data.get('warnings', [])
        if warnings:
            message += "\n<b>⚠️ Warnings</b>\n"
            for warn in warnings[:5]:
                message += f"• {warn}\n"

        message += f"\n<i>{timestamp}</i>"

        return self.send(message.strip())

    def send_signal(self, signal: Dict[str, Any]) -> bool:
        """
        시그널 알림 발송

        Args:
            signal: 시그널 딕셔너리 (type, ticker, direction, confidence 등)

        Returns:
            성공 여부
        """
        signal_type = signal.get('type', 'Unknown')
        ticker = signal.get('ticker', '')
        direction = signal.get('direction', '')
        confidence = signal.get('confidence', 0)
        description = signal.get('description', '')

        # 방향별 아이콘
        if direction.lower() in ['bullish', 'buy', 'long']:
            icon = "🟢"
            dir_text = "BULLISH"
        elif direction.lower() in ['bearish', 'sell', 'short']:
            icon = "🔴"
            dir_text = "BEARISH"
        else:
            icon = "⚪"
            dir_text = "NEUTRAL"

        message = f"""
{icon} <b>Signal: {signal_type}</b>

<b>Ticker:</b> {ticker}
<b>Direction:</b> {dir_text}
<b>Confidence:</b> {confidence:.0%}

{description}
"""
        return self.send(message.strip())


# ============================================================================
# Slack Notifier
# ============================================================================

class SlackNotifier:
    """
    Slack Webhook 알림

    사용법:
        notifier = SlackNotifier()
        notifier.send("Hello Slack!")
    """

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or SLACK_WEBHOOK_URL

    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    def send(self, message: str) -> bool:
        """메시지 발송"""
        if not self.is_configured():
            print("Slack not configured. Set SLACK_WEBHOOK_URL")
            return False

        payload = {'text': message}

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Slack send failed: {e}")
            return False

    def send_alert(self, title: str, message: str,
                   level: AlertLevel = AlertLevel.INFO) -> bool:
        """포맷된 알림 발송"""
        colors = {
            AlertLevel.INFO: "#36a64f",
            AlertLevel.WARNING: "#ffa500",
            AlertLevel.CRITICAL: "#ff0000",
        }

        payload = {
            'attachments': [{
                'color': colors.get(level, '#36a64f'),
                'title': title,
                'text': message,
                'ts': datetime.now().timestamp(),
            }]
        }

        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Slack send failed: {e}")
            return False


# ============================================================================
# Unified Notifier
# ============================================================================

class EIMASNotifier:
    """
    통합 알림 관리자

    설정된 모든 채널로 알림 발송

    사용법:
        notifier = EIMASNotifier()
        notifier.notify("중요한 알림입니다!")
        notifier.notify_alert("VIX 급등", "VIX > 30", AlertLevel.CRITICAL)
    """

    def __init__(self):
        self.telegram = TelegramNotifier()
        self.slack = SlackNotifier()

    def get_status(self) -> Dict[str, bool]:
        """채널별 설정 상태"""
        return {
            'telegram': self.telegram.is_configured(),
            'slack': self.slack.is_configured(),
        }

    def notify(self, message: str) -> Dict[str, bool]:
        """모든 채널로 메시지 발송"""
        results = {}

        if self.telegram.is_configured():
            results['telegram'] = self.telegram.send(message)

        if self.slack.is_configured():
            results['slack'] = self.slack.send(message)

        return results

    def notify_alert(self, title: str, message: str,
                     level: AlertLevel = AlertLevel.INFO) -> Dict[str, bool]:
        """모든 채널로 알림 발송"""
        results = {}

        if self.telegram.is_configured():
            results['telegram'] = self.telegram.send_alert(title, message, level)

        if self.slack.is_configured():
            results['slack'] = self.slack.send_alert(title, message, level)

        return results

    def notify_market_summary(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """시장 요약 발송"""
        results = {}

        if self.telegram.is_configured():
            results['telegram'] = self.telegram.send_market_summary(data)

        # Slack용 포맷 (별도 구현 가능)
        if self.slack.is_configured():
            # 간단히 텍스트로 변환
            text = f"Market Summary - SPY: ${data.get('spy_close', 0):,.2f}"
            results['slack'] = self.slack.send(text)

        return results


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EIMAS Notifier Test")
    print("=" * 60)

    notifier = EIMASNotifier()
    status = notifier.get_status()

    print(f"\n[Channel Status]")
    for channel, configured in status.items():
        status_text = "✅ Configured" if configured else "❌ Not configured"
        print(f"  {channel.capitalize()}: {status_text}")

    if not any(status.values()):
        print("\n⚠️ No notification channels configured!")
        print("\nTo configure Telegram:")
        print("  1. Create a bot with @BotFather (use /newbot)")
        print("  2. Get your chat ID from @userinfobot")
        print("  3. Set environment variables:")
        print("     export TELEGRAM_BOT_TOKEN='your_token'")
        print("     export TELEGRAM_CHAT_ID='your_chat_id'")
        print("\nTo configure Slack:")
        print("  1. Create an Incoming Webhook in Slack")
        print("  2. Set environment variable:")
        print("     export SLACK_WEBHOOK_URL='your_webhook_url'")
    else:
        # 테스트 메시지 발송
        print("\n[Sending test message...]")
        results = notifier.notify("🧪 EIMAS Notifier 테스트 메시지입니다.")
        for channel, success in results.items():
            status_text = "✅ Sent" if success else "❌ Failed"
            print(f"  {channel.capitalize()}: {status_text}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
