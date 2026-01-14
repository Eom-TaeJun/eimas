#!/usr/bin/env python3
"""
EIMAS Price Alerts System
=========================
Monitor price conditions and send alerts.
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import json
import os
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class AlertCondition(Enum):
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    CHANGE_PCT_UP = "change_pct_up"
    CHANGE_PCT_DOWN = "change_pct_down"
    RSI_ABOVE = "rsi_above"
    RSI_BELOW = "rsi_below"
    VOLUME_SPIKE = "volume_spike"


@dataclass
class Alert:
    id: str
    ticker: str
    condition: AlertCondition
    threshold: float
    message: str
    created_at: datetime
    triggered: bool = False
    triggered_at: Optional[datetime] = None


class AlertManager:
    """Manage price alerts"""

    def __init__(self, alerts_file: str = "data/alerts.json"):
        self.alerts_file = alerts_file
        self.alerts: List[Alert] = []
        self._load_alerts()

    def _load_alerts(self):
        """Load alerts from file"""
        if os.path.exists(self.alerts_file):
            try:
                with open(self.alerts_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        item['condition'] = AlertCondition(item['condition'])
                        item['created_at'] = datetime.fromisoformat(item['created_at'])
                        if item.get('triggered_at'):
                            item['triggered_at'] = datetime.fromisoformat(item['triggered_at'])
                        self.alerts.append(Alert(**item))
            except Exception:
                self.alerts = []

    def _save_alerts(self):
        """Save alerts to file"""
        os.makedirs(os.path.dirname(self.alerts_file), exist_ok=True)
        data = []
        for alert in self.alerts:
            d = asdict(alert)
            d['condition'] = alert.condition.value
            d['created_at'] = alert.created_at.isoformat()
            if alert.triggered_at:
                d['triggered_at'] = alert.triggered_at.isoformat()
            data.append(d)
        with open(self.alerts_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_alert(self, ticker: str, condition: AlertCondition, threshold: float, message: str = "") -> Alert:
        """Add a new alert"""
        alert = Alert(
            id=f"{ticker}_{condition.value}_{threshold}_{datetime.now().timestamp():.0f}",
            ticker=ticker.upper(),
            condition=condition,
            threshold=threshold,
            message=message or f"{ticker} {condition.value} {threshold}",
            created_at=datetime.now()
        )
        self.alerts.append(alert)
        self._save_alerts()
        return alert

    def remove_alert(self, alert_id: str):
        """Remove an alert"""
        self.alerts = [a for a in self.alerts if a.id != alert_id]
        self._save_alerts()

    def check_alerts(self) -> List[Alert]:
        """Check all alerts and return triggered ones"""
        triggered = []

        for alert in self.alerts:
            if alert.triggered:
                continue

            try:
                stock = yf.Ticker(alert.ticker)
                info = stock.info
                price = info.get('regularMarketPrice') or info.get('previousClose', 0)
                change_pct = info.get('regularMarketChangePercent', 0)

                is_triggered = False

                if alert.condition == AlertCondition.PRICE_ABOVE:
                    is_triggered = price > alert.threshold
                elif alert.condition == AlertCondition.PRICE_BELOW:
                    is_triggered = price < alert.threshold
                elif alert.condition == AlertCondition.CHANGE_PCT_UP:
                    is_triggered = change_pct > alert.threshold
                elif alert.condition == AlertCondition.CHANGE_PCT_DOWN:
                    is_triggered = change_pct < -alert.threshold
                elif alert.condition == AlertCondition.VOLUME_SPIKE:
                    vol = info.get('regularMarketVolume', 0)
                    avg_vol = info.get('averageVolume', 1)
                    is_triggered = vol > avg_vol * alert.threshold

                if is_triggered:
                    alert.triggered = True
                    alert.triggered_at = datetime.now()
                    triggered.append(alert)

            except Exception:
                continue

        self._save_alerts()
        return triggered

    def list_alerts(self):
        """Print all alerts"""
        print("\n" + "=" * 70)
        print("EIMAS Price Alerts")
        print("=" * 70)

        active = [a for a in self.alerts if not a.triggered]
        triggered = [a for a in self.alerts if a.triggered]

        if active:
            print("\nACTIVE ALERTS:")
            print("-" * 70)
            for a in active:
                print(f"  {a.ticker:<6} {a.condition.value:<15} {a.threshold:>10.2f}  {a.message}")

        if triggered:
            print("\nTRIGGERED ALERTS:")
            print("-" * 70)
            for a in triggered:
                print(f"  {a.ticker:<6} {a.condition.value:<15} at {a.triggered_at.strftime('%Y-%m-%d %H:%M')}")

        print("=" * 70)


if __name__ == "__main__":
    manager = AlertManager()

    # Example alerts
    manager.add_alert("SPY", AlertCondition.PRICE_ABOVE, 600, "SPY breaks $600")
    manager.add_alert("AAPL", AlertCondition.RSI_BELOW, 30, "AAPL oversold")
    manager.add_alert("NVDA", AlertCondition.CHANGE_PCT_DOWN, 5, "NVDA drops 5%")

    manager.list_alerts()

    # Check alerts
    triggered = manager.check_alerts()
    if triggered:
        print(f"\n{len(triggered)} alerts triggered!")
        for a in triggered:
            print(f"  - {a.message}")
