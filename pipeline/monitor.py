#!/usr/bin/env python3
"""
EIMAS Pipeline Monitor
======================
파이프라인 상태 감시, 실패 로깅, 자동 복구 메커니즘
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

LOGS_DIR = Path(__file__).parent.parent / "logs"
FAILURE_LOG = LOGS_DIR / "failures.jsonl"

logger = logging.getLogger(__name__)


class PipelineMonitor:
    """파이프라인 단계별 실패 감지 및 복구 관리"""

    MAX_RETRIES = 3

    def __init__(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 상태 확인
    # ------------------------------------------------------------------

    def check_health(self) -> dict:
        """마지막 실행 결과 확인, 비정상 상태 감지"""
        if not FAILURE_LOG.exists():
            return {"status": "healthy", "last_failure": None}

        failures = self._read_failures()
        if not failures:
            return {"status": "healthy", "last_failure": None}

        last = failures[-1]
        return {
            "status": "degraded" if last.get("resolved") else "failed",
            "last_failure": last,
            "total_failures": len(failures),
        }

    # ------------------------------------------------------------------
    # 실패 로깅
    # ------------------------------------------------------------------

    def log_failure(self, stage: str, error: str, context: Optional[dict] = None):
        """단계별 실패를 failures.jsonl 에 기록"""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "stage": stage,
            "error": str(error),
            "context": context or {},
            "resolved": False,
        }
        with FAILURE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.error(f"[{stage}] 실패 기록: {error}")

    # ------------------------------------------------------------------
    # 복구 시도
    # ------------------------------------------------------------------

    def attempt_recovery(self, stage: str, fn, *args, **kwargs):
        """
        실패한 단계를 최대 MAX_RETRIES 회 재시도.

        사용 예:
            result = monitor.attempt_recovery("macro-analysis", run_macro, data)
        """
        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.info(f"[{stage}] 복구 시도 {attempt}/{self.MAX_RETRIES}")
                result = fn(*args, **kwargs)
                self._mark_resolved(stage)
                return result
            except Exception as e:
                last_error = e
                self.log_failure(stage, str(e), {"attempt": attempt})
                if attempt < self.MAX_RETRIES:
                    time.sleep(2 ** attempt)  # 지수 백오프

        raise RuntimeError(
            f"[{stage}] {self.MAX_RETRIES}회 재시도 후 복구 실패: {last_error}"
        )

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    def _read_failures(self) -> list:
        records = []
        with FAILURE_LOG.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def _mark_resolved(self, stage: str):
        failures = self._read_failures()
        updated = []
        for r in failures:
            if r["stage"] == stage and not r["resolved"]:
                r["resolved"] = True
            updated.append(r)
        with FAILURE_LOG.open("w", encoding="utf-8") as f:
            for r in updated:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
