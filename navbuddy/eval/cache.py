"""Upstash Redis inference cache for NavBuddy OpenRouter calls.

Caches inference results by (sample_id, model_id, modality, augment) so that
repeated runs of ``navbuddy evaluate`` skip already-completed samples without
hitting the API again.

Usage::

    from navbuddy.eval.cache import InferenceCache

    cache = InferenceCache.from_env()  # reads UPSTASH_REDIS_REST_URL/TOKEN
    result = cache.get(sample_id, model_id, "video + prior", "")
    if result is None:
        result = call_openrouter(...)
        cache.set(sample_id, model_id, "video + prior", "", result)

Environment variables::

    UPSTASH_REDIS_REST_URL   – Upstash REST endpoint URL
    UPSTASH_REDIS_REST_TOKEN – Upstash REST token
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_KEY_PREFIX = "navbuddy:inference:"
_DEFAULT_TTL = 86400 * 30  # 30 days


class InferenceCache:
    """Thin wrapper around Upstash Redis for caching inference results.

    Args:
        url: Upstash Redis REST URL.
        token: Upstash Redis REST token.
        ttl_seconds: TTL for cached entries (default 30 days).
    """

    def __init__(self, url: str, token: str, ttl_seconds: int = _DEFAULT_TTL):
        try:
            from upstash_redis import Redis
        except ImportError as exc:
            raise RuntimeError(
                "InferenceCache requires upstash-redis. "
                "Install: pip install upstash-redis"
            ) from exc

        self._redis = Redis(url=url, token=token)
        self._ttl = ttl_seconds
        logger.info("InferenceCache connected to Upstash Redis (ttl=%ds)", ttl_seconds)

    @classmethod
    def from_env(cls, ttl_seconds: int = _DEFAULT_TTL) -> "InferenceCache":
        """Create from UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN env vars."""
        url = os.environ.get("UPSTASH_REDIS_REST_URL", "")
        token = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
        if not url or not token:
            raise RuntimeError(
                "Set UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN "
                "environment variables to use InferenceCache."
            )
        return cls(url=url, token=token, ttl_seconds=ttl_seconds)

    @staticmethod
    def _key(sample_id: str, model_id: str, modality: str, augment: str = "") -> str:
        raw = f"{sample_id}::{model_id}::{modality}::{augment}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:24]
        return _KEY_PREFIX + digest

    def get(
        self,
        sample_id: str,
        model_id: str,
        modality: str,
        augment: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Return cached result dict, or None on miss."""
        key = self._key(sample_id, model_id, modality, augment)
        try:
            val = self._redis.get(key)
            if val is None:
                return None
            return json.loads(val) if isinstance(val, (str, bytes)) else val
        except Exception as exc:
            logger.warning("Cache GET failed for %s: %s", sample_id, exc)
            return None

    def set(
        self,
        sample_id: str,
        model_id: str,
        modality: str,
        augment: str,
        result: Dict[str, Any],
    ) -> None:
        """Store result dict in cache."""
        key = self._key(sample_id, model_id, modality, augment)
        try:
            self._redis.setex(key, self._ttl, json.dumps(result))
        except Exception as exc:
            logger.warning("Cache SET failed for %s: %s", sample_id, exc)

    def delete(self, sample_id: str, model_id: str, modality: str, augment: str = "") -> None:
        """Explicitly delete a cached entry."""
        key = self._key(sample_id, model_id, modality, augment)
        try:
            self._redis.delete(key)
        except Exception as exc:
            logger.warning("Cache DELETE failed for %s: %s", sample_id, exc)
