#!/usr/bin/env python3
"""
EIMAS Scheduler
===============
Automated scheduling for running the EIMAS pipeline.

Usage:
    # Run once
    python scheduler.py --run-now

    # Run as daemon (every hour)
    python scheduler.py --daemon --interval 60

    # Generate cron entry
    python scheduler.py --cron

    # Systemd service
    python scheduler.py --systemd
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import os
import time
import argparse
import signal
from datetime import datetime, timedelta
from typing import Optional


class EIMASScheduler:
    """Scheduler for EIMAS pipeline"""

    def __init__(self, interval_minutes: int = 60):
        self.interval = interval_minutes * 60  # Convert to seconds
        self.running = False
        self.last_run: Optional[datetime] = None

    def run_pipeline(self) -> bool:
        """Run the full EIMAS pipeline"""
        try:
            print("\n" + "=" * 60)
            print(f"EIMAS Scheduled Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)

            # Import here to avoid circular imports
            from lib.signal_pipeline import SignalPipeline
            from lib.debate_agent import DebateAgent
            from lib.notifications import NotificationService

            # 1. Collect signals
            print("\n[1/3] Collecting signals...")
            pipeline = SignalPipeline()
            signals = pipeline.run()
            consensus = pipeline.get_consensus()
            pipeline.print_summary()

            # Convert signals to dict
            signal_dicts = []
            for s in signals:
                signal_dicts.append({
                    'source': s.source.value,
                    'action': s.action.value,
                    'ticker': s.ticker,
                    'conviction': s.conviction,
                    'reasoning': s.reasoning,
                    'metadata': s.metadata
                })

            # 2. Generate report
            print("\n[2/3] Generating report...")
            agent = DebateAgent()
            report = agent.generate_report(signal_dicts, consensus)
            filepath = agent.save_report(report)
            print(f"  Report saved: {filepath}")

            # 3. Send notifications
            print("\n[3/3] Sending notifications...")
            notifier = NotificationService()
            results = notifier.send_signal_alert(signal_dicts, consensus)

            self.last_run = datetime.now()

            print("\n" + "=" * 60)
            print("Scheduled run completed successfully!")
            print("=" * 60)

            return True

        except Exception as e:
            print(f"\nError during scheduled run: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_daemon(self):
        """Run as a daemon process"""
        print("=" * 60)
        print("EIMAS Scheduler Daemon")
        print("=" * 60)
        print(f"Interval: {self.interval // 60} minutes")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Press Ctrl+C to stop\n")

        self.running = True

        # Handle shutdown gracefully
        def handle_signal(signum, frame):
            print("\nShutdown signal received...")
            self.running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        while self.running:
            # Run pipeline
            self.run_pipeline()

            # Wait for next interval
            if self.running:
                next_run = datetime.now() + timedelta(seconds=self.interval)
                print(f"\nNext run at: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
                print("Waiting...")

                # Sleep in small increments to allow for graceful shutdown
                for _ in range(self.interval):
                    if not self.running:
                        break
                    time.sleep(1)

        print("Scheduler stopped.")

    def get_status(self) -> dict:
        """Get scheduler status"""
        return {
            'running': self.running,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'interval_minutes': self.interval // 60
        }


def generate_cron_entry(interval_minutes: int = 60) -> str:
    """Generate crontab entry"""
    script_path = os.path.abspath(__file__)
    python_path = sys.executable
    eimas_path = os.path.dirname(script_path)

    # Determine cron schedule
    if interval_minutes == 60:
        schedule = "0 * * * *"  # Every hour
    elif interval_minutes == 30:
        schedule = "0,30 * * * *"  # Every 30 minutes
    elif interval_minutes == 15:
        schedule = "*/15 * * * *"  # Every 15 minutes
    elif interval_minutes == 1440:  # Daily
        schedule = "0 8 * * *"  # 8 AM daily
    else:
        schedule = f"*/{interval_minutes} * * * *"

    cron_entry = f"""\
# EIMAS Scheduled Pipeline
# Runs every {interval_minutes} minutes
{schedule} cd {eimas_path} && PYTHONPATH={eimas_path}:$PYTHONPATH {python_path} {script_path} --run-now >> /var/log/eimas.log 2>&1
"""
    return cron_entry


def generate_systemd_service() -> str:
    """Generate systemd service file"""
    script_path = os.path.abspath(__file__)
    python_path = sys.executable
    eimas_path = os.path.dirname(script_path)
    user = os.environ.get('USER', 'root')

    service = f"""\
[Unit]
Description=EIMAS Economic Intelligence Multi-Agent System
After=network.target

[Service]
Type=simple
User={user}
WorkingDirectory={eimas_path}
Environment="PYTHONPATH={eimas_path}:$PYTHONPATH"
ExecStart={python_path} {script_path} --daemon --interval 60
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
"""
    return service


def main():
    parser = argparse.ArgumentParser(
        description="EIMAS Scheduler - Automated pipeline execution"
    )
    parser.add_argument(
        '--run-now',
        action='store_true',
        help='Run the pipeline immediately and exit'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run as a daemon process'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Interval in minutes between runs (default: 60)'
    )
    parser.add_argument(
        '--cron',
        action='store_true',
        help='Generate crontab entry'
    )
    parser.add_argument(
        '--systemd',
        action='store_true',
        help='Generate systemd service file'
    )

    args = parser.parse_args()

    if args.cron:
        print("Add this to your crontab (crontab -e):\n")
        print(generate_cron_entry(args.interval))
        return

    if args.systemd:
        print("Save this as /etc/systemd/system/eimas.service:\n")
        print(generate_systemd_service())
        print("\nThen run:")
        print("  sudo systemctl daemon-reload")
        print("  sudo systemctl enable eimas")
        print("  sudo systemctl start eimas")
        return

    scheduler = EIMASScheduler(interval_minutes=args.interval)

    if args.run_now:
        success = scheduler.run_pipeline()
        sys.exit(0 if success else 1)
    elif args.daemon:
        scheduler.run_daemon()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
