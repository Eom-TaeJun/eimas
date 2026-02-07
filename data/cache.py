#!/usr/bin/env python3
"""
EIMAS Cache System
===================
Redis 및 로컬 캐싱 시스템

주요 기능:
1. Redis 캐싱 (분산 환경)
2. 로컬 파일 캐싱 (폴백)
3. 메모리 캐싱 (LRU)
4. 캐시 무효화 전략

Usage:
    from data.cache import CacheManager, cache

    # 데코레이터 사용
    @cache(ttl=3600)
    def expensive_function():
        ...

    # 직접 사용
    cache_mgr = CacheManager()
    cache_mgr.set('key', data, ttl=3600)
    data = cache_mgr.get('key')
"""

import json
import pickle
import hashlib
import functools
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from collections import OrderedDict
import threading
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / "data" / "cache"
DEFAULT_TTL = 3600  # 1시간
MAX_MEMORY_ITEMS = 1000  # 메모리 캐시 최대 항목

# Redis 설정 (있으면 사용, 없으면 로컬 캐시)
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'prefix': 'eimas:',
}


# ============================================================================
# Enums and Data Classes
# ============================================================================

class CacheBackend(str, Enum):
    """캐시 백엔드"""
    REDIS = 'redis'
    FILE = 'file'
    MEMORY = 'memory'


class CachePolicy(str, Enum):
    """캐시 정책"""
    LRU = 'lru'           # Least Recently Used
    LFU = 'lfu'           # Least Frequently Used
    FIFO = 'fifo'         # First In First Out
    TTL = 'ttl'           # Time To Live only


@dataclass
class CacheStats:
    """캐시 통계"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'sets': self.sets,
            'deletes': self.deletes,
            'evictions': self.evictions,
            'hit_rate': f"{self.hit_rate:.1%}",
        }


@dataclass
class CacheEntry:
    """캐시 엔트리"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None


# ============================================================================
# Memory Cache (LRU)
# ============================================================================

class LRUCache:
    """LRU 메모리 캐시"""

    def __init__(self, max_size: int = MAX_MEMORY_ITEMS):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.RLock()
        self.stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        """캐시 조회"""
        with self.lock:
            if key not in self.cache:
                self.stats.misses += 1
                return None

            entry = self.cache[key]

            # TTL 확인
            if entry.expires_at and datetime.now() > entry.expires_at:
                del self.cache[key]
                self.stats.misses += 1
                return None

            # LRU 업데이트 - 최근 사용을 맨 뒤로
            self.cache.move_to_end(key)
            entry.access_count += 1
            entry.last_accessed = datetime.now()

            self.stats.hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """캐시 저장"""
        with self.lock:
            # 용량 초과시 가장 오래된 항목 제거
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats.evictions += 1

            expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
            )

            self.cache[key] = entry
            self.cache.move_to_end(key)
            self.stats.sets += 1

            return True

    def delete(self, key: str) -> bool:
        """캐시 삭제"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats.deletes += 1
                return True
            return False

    def clear(self):
        """캐시 초기화"""
        with self.lock:
            self.cache.clear()

    def keys(self) -> List[str]:
        """모든 키 조회"""
        with self.lock:
            return list(self.cache.keys())

    def size(self) -> int:
        """캐시 크기"""
        return len(self.cache)


# ============================================================================
# File Cache
# ============================================================================

class FileCache:
    """파일 기반 캐시"""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = CacheStats()
        self.lock = threading.RLock()

    def _get_path(self, key: str) -> Path:
        """캐시 파일 경로"""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.cache"

    def _get_meta_path(self, key: str) -> Path:
        """메타데이터 파일 경로"""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.meta"

    def get(self, key: str) -> Optional[Any]:
        """캐시 조회"""
        with self.lock:
            path = self._get_path(key)
            meta_path = self._get_meta_path(key)

            if not path.exists():
                self.stats.misses += 1
                return None

            try:
                # 메타데이터 확인
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                        expires_at = datetime.fromisoformat(meta['expires_at']) if meta.get('expires_at') else None
                        if expires_at and datetime.now() > expires_at:
                            self.delete(key)
                            self.stats.misses += 1
                            return None

                # 데이터 로드
                with open(path, 'rb') as f:
                    data = pickle.load(f)

                self.stats.hits += 1
                return data

            except Exception as e:
                logger.warning(f"Cache read error: {e}")
                self.stats.misses += 1
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """캐시 저장"""
        with self.lock:
            path = self._get_path(key)
            meta_path = self._get_meta_path(key)

            try:
                # 데이터 저장
                with open(path, 'wb') as f:
                    pickle.dump(value, f)

                # 메타데이터 저장
                meta = {
                    'key': key,
                    'created_at': datetime.now().isoformat(),
                    'expires_at': (datetime.now() + timedelta(seconds=ttl)).isoformat() if ttl else None,
                }
                with open(meta_path, 'w') as f:
                    json.dump(meta, f)

                self.stats.sets += 1
                return True

            except Exception as e:
                logger.warning(f"Cache write error: {e}")
                return False

    def delete(self, key: str) -> bool:
        """캐시 삭제"""
        with self.lock:
            path = self._get_path(key)
            meta_path = self._get_meta_path(key)

            deleted = False
            if path.exists():
                path.unlink()
                deleted = True
            if meta_path.exists():
                meta_path.unlink()

            if deleted:
                self.stats.deletes += 1
            return deleted

    def clear(self):
        """캐시 초기화"""
        with self.lock:
            for f in self.cache_dir.glob('*.cache'):
                f.unlink()
            for f in self.cache_dir.glob('*.meta'):
                f.unlink()

    def keys(self) -> List[str]:
        """모든 키 조회 (해시된 키)"""
        return [f.stem for f in self.cache_dir.glob('*.cache')]

    def size(self) -> int:
        """캐시 파일 수"""
        return len(list(self.cache_dir.glob('*.cache')))

    def cleanup_expired(self) -> int:
        """만료된 캐시 정리"""
        cleaned = 0
        for meta_path in self.cache_dir.glob('*.meta'):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    expires_at = datetime.fromisoformat(meta['expires_at']) if meta.get('expires_at') else None
                    if expires_at and datetime.now() > expires_at:
                        cache_path = meta_path.with_suffix('.cache')
                        if cache_path.exists():
                            cache_path.unlink()
                        meta_path.unlink()
                        cleaned += 1
            except:
                pass
        return cleaned


# ============================================================================
# Redis Cache
# ============================================================================

class RedisCache:
    """Redis 캐시"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or REDIS_CONFIG
        self.prefix = self.config.get('prefix', 'eimas:')
        self.stats = CacheStats()
        self.client = None

        self._connect()

    def _connect(self) -> bool:
        """Redis 연결"""
        try:
            import redis
            self.client = redis.Redis(
                host=self.config['host'],
                port=self.config['port'],
                db=self.config['db'],
                decode_responses=False,
            )
            self.client.ping()
            logger.info("Connected to Redis")
            return True
        except ImportError:
            logger.warning("redis-py not installed, falling back to file cache")
            return False
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        """연결 상태"""
        if not self.client:
            return False
        try:
            self.client.ping()
            return True
        except:
            return False

    def _make_key(self, key: str) -> str:
        """Redis 키 생성"""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """캐시 조회"""
        if not self.is_connected:
            self.stats.misses += 1
            return None

        try:
            data = self.client.get(self._make_key(key))
            if data is None:
                self.stats.misses += 1
                return None

            self.stats.hits += 1
            return pickle.loads(data)

        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            self.stats.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """캐시 저장"""
        if not self.is_connected:
            return False

        try:
            data = pickle.dumps(value)
            if ttl:
                self.client.setex(self._make_key(key), ttl, data)
            else:
                self.client.set(self._make_key(key), data)

            self.stats.sets += 1
            return True

        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """캐시 삭제"""
        if not self.is_connected:
            return False

        try:
            result = self.client.delete(self._make_key(key))
            if result:
                self.stats.deletes += 1
            return bool(result)
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return False

    def clear(self):
        """캐시 초기화 (prefix 기준)"""
        if not self.is_connected:
            return

        try:
            keys = self.client.keys(f"{self.prefix}*")
            if keys:
                self.client.delete(*keys)
        except Exception as e:
            logger.warning(f"Redis clear error: {e}")

    def keys(self) -> List[str]:
        """모든 키 조회"""
        if not self.is_connected:
            return []

        try:
            keys = self.client.keys(f"{self.prefix}*")
            return [k.decode().replace(self.prefix, '') for k in keys]
        except:
            return []

    def size(self) -> int:
        """캐시 크기"""
        return len(self.keys())


# ============================================================================
# Cache Manager (통합)
# ============================================================================

class CacheManager:
    """통합 캐시 매니저"""

    def __init__(
        self,
        backend: CacheBackend = CacheBackend.FILE,
        use_memory: bool = True,
        fallback: bool = True,
    ):
        self.use_memory = use_memory
        self.fallback = fallback

        # 메모리 캐시 (항상 1차)
        self.memory_cache = LRUCache() if use_memory else None

        # 백엔드 초기화
        self.backend = backend
        if backend == CacheBackend.REDIS:
            self.primary_cache = RedisCache()
            if not self.primary_cache.is_connected and fallback:
                logger.info("Falling back to file cache")
                self.primary_cache = FileCache()
                self.backend = CacheBackend.FILE
        elif backend == CacheBackend.FILE:
            self.primary_cache = FileCache()
        else:
            self.primary_cache = None  # 메모리만 사용

    def get(self, key: str) -> Optional[Any]:
        """캐시 조회"""
        # 1. 메모리 캐시 확인
        if self.memory_cache:
            value = self.memory_cache.get(key)
            if value is not None:
                return value

        # 2. 백엔드 캐시 확인
        if self.primary_cache:
            value = self.primary_cache.get(key)
            if value is not None:
                # 메모리 캐시에 승격
                if self.memory_cache:
                    self.memory_cache.set(key, value)
                return value

        return None

    def set(self, key: str, value: Any, ttl: int = DEFAULT_TTL) -> bool:
        """캐시 저장"""
        success = True

        # 1. 메모리 캐시
        if self.memory_cache:
            success = self.memory_cache.set(key, value, ttl) and success

        # 2. 백엔드 캐시
        if self.primary_cache:
            success = self.primary_cache.set(key, value, ttl) and success

        return success

    def delete(self, key: str) -> bool:
        """캐시 삭제"""
        success = True

        if self.memory_cache:
            self.memory_cache.delete(key)

        if self.primary_cache:
            success = self.primary_cache.delete(key)

        return success

    def clear(self):
        """캐시 초기화"""
        if self.memory_cache:
            self.memory_cache.clear()

        if self.primary_cache:
            self.primary_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        stats = {
            'backend': self.backend.value,
        }

        if self.memory_cache:
            stats['memory'] = {
                **self.memory_cache.stats.to_dict(),
                'size': self.memory_cache.size(),
            }

        if self.primary_cache:
            stats['primary'] = {
                **self.primary_cache.stats.to_dict(),
                'size': self.primary_cache.size(),
            }

        return stats

    def info(self) -> str:
        """캐시 정보"""
        stats = self.get_stats()
        lines = [
            "=== Cache Info ===",
            f"Backend: {stats['backend']}",
        ]

        if 'memory' in stats:
            lines.append(f"Memory Cache: {stats['memory']['size']} items, {stats['memory']['hit_rate']} hit rate")

        if 'primary' in stats:
            lines.append(f"Primary Cache: {stats['primary']['size']} items, {stats['primary']['hit_rate']} hit rate")

        return "\n".join(lines)


# ============================================================================
# Decorator
# ============================================================================

# 전역 캐시 매니저
_global_cache: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """전역 캐시 매니저 조회"""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache


def cache(
    ttl: int = DEFAULT_TTL,
    key_prefix: str = '',
    cache_none: bool = False,
):
    """캐시 데코레이터

    Usage:
        @cache(ttl=3600)
        def expensive_function(arg1, arg2):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 캐시 키 생성
            key_parts = [key_prefix or func.__name__]
            key_parts.extend(str(a) for a in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ':'.join(key_parts)

            # 캐시 조회
            cache_mgr = get_cache()
            cached = cache_mgr.get(cache_key)
            if cached is not None or (cache_none and cached is None):
                return cached

            # 함수 실행
            result = func(*args, **kwargs)

            # 캐시 저장
            if result is not None or cache_none:
                cache_mgr.set(cache_key, result, ttl)

            return result

        return wrapper
    return decorator


# ============================================================================
# Convenience Functions
# ============================================================================

def cached_fetch(key: str, fetch_func: Callable, ttl: int = DEFAULT_TTL) -> Any:
    """캐시된 데이터 fetch"""
    cache_mgr = get_cache()

    # 캐시 확인
    cached = cache_mgr.get(key)
    if cached is not None:
        return cached

    # Fetch
    data = fetch_func()

    # 캐시 저장
    if data is not None:
        cache_mgr.set(key, data, ttl)

    return data


def invalidate_pattern(pattern: str):
    """패턴에 맞는 캐시 무효화"""
    cache_mgr = get_cache()

    if hasattr(cache_mgr.primary_cache, 'keys'):
        keys = cache_mgr.primary_cache.keys()
        for key in keys:
            if pattern in key:
                cache_mgr.delete(key)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Testing Cache System...")

    # 1. 메모리 캐시 테스트
    print("\n=== Memory Cache ===")
    mem_cache = LRUCache(max_size=5)
    for i in range(7):
        mem_cache.set(f"key_{i}", f"value_{i}")
    print(f"Size: {mem_cache.size()}")
    print(f"Keys: {mem_cache.keys()}")
    print(f"Stats: {mem_cache.stats.to_dict()}")

    # 2. 파일 캐시 테스트
    print("\n=== File Cache ===")
    file_cache = FileCache()
    file_cache.set("test_key", {"data": [1, 2, 3]}, ttl=60)
    result = file_cache.get("test_key")
    print(f"Get result: {result}")
    print(f"Stats: {file_cache.stats.to_dict()}")

    # 3. 캐시 매니저 테스트
    print("\n=== Cache Manager ===")
    cache_mgr = CacheManager(backend=CacheBackend.FILE)
    cache_mgr.set("manager_test", [1, 2, 3, 4, 5], ttl=300)
    result = cache_mgr.get("manager_test")
    print(f"Manager get: {result}")
    print(cache_mgr.info())

    # 4. 데코레이터 테스트
    print("\n=== Decorator Test ===")

    @cache(ttl=60)
    def slow_function(x, y):
        print(f"  Computing {x} + {y}...")
        return x + y

    print("First call:")
    result1 = slow_function(1, 2)
    print(f"  Result: {result1}")

    print("Second call (cached):")
    result2 = slow_function(1, 2)
    print(f"  Result: {result2}")

    print("\n=== Final Stats ===")
    print(get_cache().info())
