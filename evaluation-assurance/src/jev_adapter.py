"""
Optional: TypeSafe's Jev as an extra scorer. NOT used for any result in this repo.

Only worth it for evidence that isn't already a number (free-text tool output,
error messages). Uses the official SDK (pip install typesafe-sdk, needs
TYPESAFE_API_KEY), not third-party wrappers.

It must never block the pipeline: on 429 honour retry_after, otherwise back
off exponentially; after repeated failures the circuit breaker opens and we
fall back to the in-house score. Only send redacted trace summaries.
"""

import os
import random
import time


class CircuitBreaker:
    def __init__(self, fail_threshold=5, cooldown_s=60):
        self.fail_threshold, self.cooldown_s = fail_threshold, cooldown_s
        self.failures, self.opened_at = 0, None

    def allow(self):
        if self.opened_at is None:
            return True
        if time.time() - self.opened_at > self.cooldown_s:
            self.opened_at, self.failures = None, 0
            return True
        return False

    def record(self, ok: bool):
        if ok:
            self.failures = 0
        else:
            self.failures += 1
            if self.failures >= self.fail_threshold:
                self.opened_at = time.time()


class JevScorer:
    # returns P(task truly succeeded), or None -> caller uses in-house score

    QUESTION = ("Given this agent execution summary, did the agent truly complete the task "
                "(correct state changes, correct tool arguments, correct ordering, nothing skipped)?")

    def __init__(self, max_attempts=4, base_delay_s=0.5, breaker=None):
        from typesafe_sdk import TypeSafeClient  # optional dependency
        if not os.environ.get("TYPESAFE_API_KEY"):
            raise RuntimeError("Set TYPESAFE_API_KEY to use the optional Jev scorer.")
        self.client = TypeSafeClient()
        self.max_attempts, self.base_delay_s = max_attempts, base_delay_s
        self.breaker = breaker or CircuitBreaker()

    def score(self, trace_summary: dict):
        from typesafe_sdk import Noul, TypeSafeRateLimitError, TypeSafeError
        if not self.breaker.allow():
            return None
        for attempt in range(self.max_attempts):
            try:
                result = self.client.system_one(
                    trace_summary, {"truly_passed": Noul(instructions=self.QUESTION)})
                self.breaker.record(True)
                return float(result.nouls["truly_passed"].noul)
            except TypeSafeRateLimitError as e:
                wait = (e.retry_after_ms / 1000) if getattr(e, "retry_after_ms", None) \
                    else self.base_delay_s * 2 ** attempt
                time.sleep(wait + random.uniform(0, 0.25))
            except TypeSafeError:
                time.sleep(self.base_delay_s * 2 ** attempt + random.uniform(0, 0.25))
        self.breaker.record(False)
        return None  # caller falls back to the in-house score


def blended_score(inhouse_p: float, jev_p, weight=0.3):
    # weight should be fit on the canary set, 0.3 is just a placeholder
    return inhouse_p if jev_p is None else (1 - weight) * inhouse_p + weight * jev_p
