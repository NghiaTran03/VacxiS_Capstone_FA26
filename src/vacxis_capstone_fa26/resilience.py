"""Bounded retry policy and atomic, fingerprinted checkpoints."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

@dataclass(frozen=True)
class RetryPolicy:
    retries: int = 5
    initial: float = 1.0
    maximum: float = 60.0

    def __post_init__(self):
        if self.retries < 0 or any(not math.isfinite(x) or x < 0 for x in (self.initial, self.maximum)):
            raise ValueError("Retry counts and delays must be finite and nonnegative.")

    def delay(self, attempt):
        return min(self.maximum, self.initial * 2 ** min(attempt, 30)) * random.uniform(0.8, 1.0)

    @staticmethod
    def transient(exc):
        from openai import APIConnectionError, APIStatusError
        return isinstance(exc, APIConnectionError) or (
            isinstance(exc, APIStatusError) and (exc.status_code in (408, 409, 429) or exc.status_code >= 500)
        )

    def run(self, call):
        for attempt in range(self.retries + 1):
            try:
                return call()
            except Exception as exc:
                if attempt == self.retries or not self.transient(exc):
                    raise
                time.sleep(self.delay(attempt))

    async def arun(self, call):
        for attempt in range(self.retries + 1):
            try:
                return await call()
            except Exception as exc:
                if attempt == self.retries or not self.transient(exc):
                    raise
                await asyncio.sleep(self.delay(attempt))

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

def add_retry_arguments(parser):
    parser.add_argument("--retries", type=int, default=5, help="Additional transient API attempts")
    parser.add_argument("--retry-initial", type=float, default=1.0, help="Initial exponential backoff in seconds")
    parser.add_argument("--retry-max", type=float, default=60.0, help="Maximum backoff in seconds")

def retry_from_args(args):
    return RetryPolicy(args.retries, args.retry_initial, args.retry_max)
