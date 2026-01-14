#!/usr/bin/env python3
"""
EIMAS Health Check
===================
시스템 상태 모니터링

주요 기능:
1. 컴포넌트 상태 확인
2. 데이터 소스 연결 테스트
3. 성능 메트릭 수집
4. 알림 임계값 모니터링

Usage:
    from core.health_check import HealthChecker, check_system_health

    checker = HealthChecker()
    status = checker.check_all()
    print(status)
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import os
import psutil
import sqlite3
import socket
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging
import json

# 로깅
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# 경로
BASE_DIR = Path('/home/tj/projects/autoai/eimas')
DATA_DIR = BASE_DIR / 'data'
LOGS_DIR = BASE_DIR / 'logs'
CACHE_DIR = DATA_DIR / 'cache'
DB_PATH = DATA_DIR / 'eimas_data.db'

# 임계값
THRESHOLDS = {
    'cpu_percent': 80.0,
    'memory_percent': 85.0,
    'disk_percent': 90.0,
    'cache_age_hours': 24,
    'log_size_mb': 100,
    'db_size_mb': 500,
}


# ============================================================================
# Enums and Data Classes
# ============================================================================

class HealthStatus(str, Enum):
    """상태"""
    HEALTHY = 'healthy'
    DEGRADED = 'degraded'
    UNHEALTHY = 'unhealthy'
    UNKNOWN = 'unknown'


class ComponentType(str, Enum):
    """컴포넌트 타입"""
    SYSTEM = 'system'
    DATABASE = 'database'
    CACHE = 'cache'
    API = 'api'
    DATA_SOURCE = 'data_source'
    SCHEDULER = 'scheduler'


@dataclass
class ComponentHealth:
    """컴포넌트 상태"""
    name: str
    type: ComponentType
    status: HealthStatus
    message: str
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.now)


@dataclass
class SystemMetrics:
    """시스템 메트릭"""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    uptime_hours: float
    load_average: tuple


@dataclass
class HealthReport:
    """전체 상태 리포트"""
    timestamp: datetime
    overall_status: HealthStatus
    system_metrics: SystemMetrics
    components: List[ComponentHealth]
    issues: List[str]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        for comp in data['components']:
            comp['checked_at'] = self.timestamp.isoformat()
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


# ============================================================================
# Health Checker
# ============================================================================

class HealthChecker:
    """상태 체커"""

    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or THRESHOLDS
        self.components: List[ComponentHealth] = []
        self.issues: List[str] = []

    def check_all(self) -> HealthReport:
        """전체 상태 확인"""
        self.components = []
        self.issues = []

        print(f"\n{'='*50}")
        print("EIMAS Health Check")
        print('='*50)

        # 시스템 메트릭
        print("\n[1/5] System Metrics...")
        metrics = self._get_system_metrics()
        self._check_system_thresholds(metrics)

        # 데이터베이스
        print("[2/5] Database...")
        self._check_database()

        # 캐시
        print("[3/5] Cache...")
        self._check_cache()

        # 데이터 소스
        print("[4/5] Data Sources...")
        self._check_data_sources()

        # 로그
        print("[5/5] Logs...")
        self._check_logs()

        # 전체 상태 결정
        overall = self._determine_overall_status()

        # 요약
        summary = self._generate_summary(metrics, overall)

        report = HealthReport(
            timestamp=datetime.now(),
            overall_status=overall,
            system_metrics=metrics,
            components=self.components,
            issues=self.issues,
            summary=summary,
        )

        print("\n" + summary)
        print('='*50)

        return report

    # -------------------------------------------------------------------------
    # System Metrics
    # -------------------------------------------------------------------------

    def _get_system_metrics(self) -> SystemMetrics:
        """시스템 메트릭 수집"""
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # 업타임
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = (datetime.now() - boot_time).total_seconds() / 3600

        # Load average (Unix only)
        try:
            load = os.getloadavg()
        except:
            load = (0, 0, 0)

        return SystemMetrics(
            cpu_percent=cpu,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_percent=disk.percent,
            disk_used_gb=disk.used / (1024**3),
            disk_total_gb=disk.total / (1024**3),
            uptime_hours=uptime,
            load_average=load,
        )

    def _check_system_thresholds(self, metrics: SystemMetrics):
        """시스템 임계값 확인"""
        status = HealthStatus.HEALTHY
        issues = []

        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            issues.append(f"High CPU: {metrics.cpu_percent:.1f}%")
            status = HealthStatus.DEGRADED

        if metrics.memory_percent > self.thresholds['memory_percent']:
            issues.append(f"High Memory: {metrics.memory_percent:.1f}%")
            status = HealthStatus.DEGRADED

        if metrics.disk_percent > self.thresholds['disk_percent']:
            issues.append(f"Low Disk: {100 - metrics.disk_percent:.1f}% free")
            status = HealthStatus.UNHEALTHY

        self.components.append(ComponentHealth(
            name='system',
            type=ComponentType.SYSTEM,
            status=status,
            message=', '.join(issues) if issues else 'System healthy',
            details={
                'cpu': f"{metrics.cpu_percent:.1f}%",
                'memory': f"{metrics.memory_percent:.1f}%",
                'disk': f"{metrics.disk_percent:.1f}%",
            }
        ))

        self.issues.extend(issues)

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------

    def _check_database(self):
        """데이터베이스 상태 확인"""
        import time
        start = time.time()

        try:
            if not DB_PATH.exists():
                self.components.append(ComponentHealth(
                    name='database',
                    type=ComponentType.DATABASE,
                    status=HealthStatus.UNHEALTHY,
                    message='Database file not found',
                ))
                self.issues.append("Database file not found")
                return

            conn = sqlite3.connect(DB_PATH, timeout=5)
            cursor = conn.cursor()

            # 테이블 목록
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            # DB 크기
            db_size_mb = DB_PATH.stat().st_size / (1024**2)

            latency = (time.time() - start) * 1000
            conn.close()

            status = HealthStatus.HEALTHY
            issues = []

            if db_size_mb > self.thresholds['db_size_mb']:
                issues.append(f"Large DB: {db_size_mb:.1f}MB")
                status = HealthStatus.DEGRADED

            self.components.append(ComponentHealth(
                name='database',
                type=ComponentType.DATABASE,
                status=status,
                message=f"{len(tables)} tables, {db_size_mb:.1f}MB",
                latency_ms=latency,
                details={'tables': tables, 'size_mb': db_size_mb},
            ))

            self.issues.extend(issues)

        except Exception as e:
            self.components.append(ComponentHealth(
                name='database',
                type=ComponentType.DATABASE,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            ))
            self.issues.append(f"Database error: {e}")

    # -------------------------------------------------------------------------
    # Cache
    # -------------------------------------------------------------------------

    def _check_cache(self):
        """캐시 상태 확인"""
        try:
            if not CACHE_DIR.exists():
                self.components.append(ComponentHealth(
                    name='cache',
                    type=ComponentType.CACHE,
                    status=HealthStatus.DEGRADED,
                    message='Cache directory not found',
                ))
                return

            cache_files = list(CACHE_DIR.glob('*.pkl')) + list(CACHE_DIR.glob('*.cache'))
            cache_size = sum(f.stat().st_size for f in cache_files) / (1024**2)

            # 가장 오래된 캐시
            if cache_files:
                oldest = min(f.stat().st_mtime for f in cache_files)
                oldest_age = (datetime.now() - datetime.fromtimestamp(oldest)).total_seconds() / 3600
            else:
                oldest_age = 0

            status = HealthStatus.HEALTHY
            issues = []

            if oldest_age > self.thresholds['cache_age_hours']:
                issues.append(f"Stale cache: {oldest_age:.1f}h old")
                status = HealthStatus.DEGRADED

            self.components.append(ComponentHealth(
                name='cache',
                type=ComponentType.CACHE,
                status=status,
                message=f"{len(cache_files)} files, {cache_size:.1f}MB",
                details={
                    'file_count': len(cache_files),
                    'size_mb': cache_size,
                    'oldest_hours': oldest_age,
                },
            ))

            self.issues.extend(issues)

        except Exception as e:
            self.components.append(ComponentHealth(
                name='cache',
                type=ComponentType.CACHE,
                status=HealthStatus.UNKNOWN,
                message=str(e),
            ))

    # -------------------------------------------------------------------------
    # Data Sources
    # -------------------------------------------------------------------------

    def _check_data_sources(self):
        """데이터 소스 연결 확인"""
        import time

        # yfinance 테스트
        start = time.time()
        try:
            import yfinance as yf
            data = yf.download('SPY', period='1d', progress=False)
            latency = (time.time() - start) * 1000

            if data.empty:
                status = HealthStatus.DEGRADED
                message = "No data returned"
            else:
                status = HealthStatus.HEALTHY
                message = f"Connected ({latency:.0f}ms)"

            self.components.append(ComponentHealth(
                name='yfinance',
                type=ComponentType.DATA_SOURCE,
                status=status,
                message=message,
                latency_ms=latency,
            ))

        except Exception as e:
            self.components.append(ComponentHealth(
                name='yfinance',
                type=ComponentType.DATA_SOURCE,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            ))
            self.issues.append(f"yfinance error: {e}")

        # Redis 테스트 (옵션)
        try:
            import redis
            start = time.time()
            r = redis.Redis(host='localhost', port=6379, socket_timeout=2)
            r.ping()
            latency = (time.time() - start) * 1000

            self.components.append(ComponentHealth(
                name='redis',
                type=ComponentType.CACHE,
                status=HealthStatus.HEALTHY,
                message=f"Connected ({latency:.0f}ms)",
                latency_ms=latency,
            ))

        except ImportError:
            pass  # redis not installed
        except Exception:
            self.components.append(ComponentHealth(
                name='redis',
                type=ComponentType.CACHE,
                status=HealthStatus.DEGRADED,
                message="Not available (using file cache)",
            ))

    # -------------------------------------------------------------------------
    # Logs
    # -------------------------------------------------------------------------

    def _check_logs(self):
        """로그 상태 확인"""
        try:
            if not LOGS_DIR.exists():
                self.components.append(ComponentHealth(
                    name='logs',
                    type=ComponentType.SYSTEM,
                    status=HealthStatus.DEGRADED,
                    message='Log directory not found',
                ))
                return

            log_files = list(LOGS_DIR.glob('*.log'))
            total_size = sum(f.stat().st_size for f in log_files) / (1024**2)

            status = HealthStatus.HEALTHY
            issues = []

            if total_size > self.thresholds['log_size_mb']:
                issues.append(f"Large logs: {total_size:.1f}MB")
                status = HealthStatus.DEGRADED

            # 최근 에러 확인
            error_log = LOGS_DIR / 'error.log'
            recent_errors = 0
            if error_log.exists():
                # 최근 1시간 에러 수
                cutoff = datetime.now() - timedelta(hours=1)
                try:
                    with open(error_log, 'r') as f:
                        for line in f:
                            try:
                                entry = json.loads(line)
                                ts = datetime.fromisoformat(entry['timestamp'])
                                if ts > cutoff:
                                    recent_errors += 1
                            except:
                                pass
                except:
                    pass

            if recent_errors > 10:
                issues.append(f"High error rate: {recent_errors}/hour")
                status = HealthStatus.DEGRADED

            self.components.append(ComponentHealth(
                name='logs',
                type=ComponentType.SYSTEM,
                status=status,
                message=f"{len(log_files)} files, {total_size:.1f}MB",
                details={
                    'file_count': len(log_files),
                    'size_mb': total_size,
                    'recent_errors': recent_errors,
                },
            ))

            self.issues.extend(issues)

        except Exception as e:
            self.components.append(ComponentHealth(
                name='logs',
                type=ComponentType.SYSTEM,
                status=HealthStatus.UNKNOWN,
                message=str(e),
            ))

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def _determine_overall_status(self) -> HealthStatus:
        """전체 상태 결정"""
        statuses = [c.status for c in self.components]

        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif HealthStatus.UNKNOWN in statuses:
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY

    def _generate_summary(self, metrics: SystemMetrics, overall: HealthStatus) -> str:
        """요약 생성"""
        status_emoji = {
            HealthStatus.HEALTHY: '✅',
            HealthStatus.DEGRADED: '⚠️',
            HealthStatus.UNHEALTHY: '❌',
            HealthStatus.UNKNOWN: '❓',
        }

        lines = [
            f"\n=== Health Report ===",
            f"Status: {status_emoji[overall]} {overall.value.upper()}",
            "",
            "System:",
            f"  CPU: {metrics.cpu_percent:.1f}%",
            f"  Memory: {metrics.memory_percent:.1f}% ({metrics.memory_used_gb:.1f}/{metrics.memory_total_gb:.1f}GB)",
            f"  Disk: {metrics.disk_percent:.1f}% ({metrics.disk_used_gb:.1f}/{metrics.disk_total_gb:.1f}GB)",
            "",
            "Components:",
        ]

        for comp in self.components:
            emoji = status_emoji[comp.status]
            latency = f" ({comp.latency_ms:.0f}ms)" if comp.latency_ms else ""
            lines.append(f"  {emoji} {comp.name}: {comp.message}{latency}")

        if self.issues:
            lines.append("")
            lines.append("Issues:")
            for issue in self.issues:
                lines.append(f"  - {issue}")

        return "\n".join(lines)


# ============================================================================
# Quick Checks
# ============================================================================

def check_system_health() -> HealthReport:
    """빠른 시스템 상태 확인"""
    checker = HealthChecker()
    return checker.check_all()


def is_healthy() -> bool:
    """시스템 정상 여부"""
    report = check_system_health()
    return report.overall_status == HealthStatus.HEALTHY


def get_metrics() -> SystemMetrics:
    """시스템 메트릭 조회"""
    checker = HealthChecker()
    return checker._get_system_metrics()


def check_component(name: str) -> ComponentHealth:
    """특정 컴포넌트 확인"""
    checker = HealthChecker()
    report = checker.check_all()

    for comp in report.components:
        if comp.name == name:
            return comp

    return ComponentHealth(
        name=name,
        type=ComponentType.SYSTEM,
        status=HealthStatus.UNKNOWN,
        message="Component not found",
    )


# ============================================================================
# API Endpoint Helper
# ============================================================================

def get_health_json() -> str:
    """API용 JSON 상태"""
    report = check_system_health()
    return report.to_json()


def get_ready_status() -> Dict[str, bool]:
    """준비 상태 (k8s readiness probe용)"""
    report = check_system_health()
    return {
        'ready': report.overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED],
        'status': report.overall_status.value,
    }


def get_live_status() -> Dict[str, bool]:
    """생존 상태 (k8s liveness probe용)"""
    report = check_system_health()
    return {
        'alive': report.overall_status != HealthStatus.UNHEALTHY,
        'status': report.overall_status.value,
    }


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    # 전체 상태 확인
    report = check_system_health()

    print("\n\n=== Component Details ===")
    for comp in report.components:
        print(f"\n{comp.name} ({comp.type.value}):")
        print(f"  Status: {comp.status.value}")
        print(f"  Message: {comp.message}")
        if comp.details:
            for k, v in comp.details.items():
                print(f"  {k}: {v}")

    # JSON 출력
    print("\n\n=== JSON Output ===")
    print(report.to_json()[:500] + "...")

    # 빠른 체크
    print("\n\n=== Quick Checks ===")
    print(f"Is Healthy: {is_healthy()}")
    print(f"Ready: {get_ready_status()}")
    print(f"Live: {get_live_status()}")
