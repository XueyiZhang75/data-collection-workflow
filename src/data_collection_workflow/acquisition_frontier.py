"""Persistent session acquisition targets, independent of HTTP attempt billing."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import sqlite3
import time

from .acquisition_budget import finite_priority, nonnegative_integer


class AcquisitionFrontier:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS acquisition_frontier (
                position INTEGER PRIMARY KEY, target_id TEXT UNIQUE NOT NULL,
                url TEXT NOT NULL, source_id TEXT NOT NULL, priority REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                payload TEXT NOT NULL, reason TEXT, budget_revision INTEGER NOT NULL DEFAULT 0,
                updated REAL NOT NULL, result_ref TEXT, active_strategy TEXT,
                strategy_attempts TEXT NOT NULL DEFAULT '{}')""")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(acquisition_frontier)")}
            if "result_ref" not in columns:
                db.execute("ALTER TABLE acquisition_frontier ADD COLUMN result_ref TEXT")
            if "active_strategy" not in columns:
                db.execute("ALTER TABLE acquisition_frontier ADD COLUMN active_strategy TEXT")
            if "strategy_attempts" not in columns:
                db.execute("ALTER TABLE acquisition_frontier ADD COLUMN strategy_attempts TEXT NOT NULL DEFAULT '{}'")
                # Pre-versioned queue history cannot safely be assigned a new
                # strategy. Preserve it conservatively against every strategy.
                for row in db.execute("SELECT position,attempts FROM acquisition_frontier WHERE attempts>0").fetchall():
                    db.execute("UPDATE acquisition_frontier SET strategy_attempts=? WHERE position=?",
                               (json.dumps({"legacy_unknown": row["attempts"]}), row["position"]))

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=60)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _row(row):
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        result["strategy_attempts"] = json.loads(result["strategy_attempts"])
        return result

    @staticmethod
    def _strategy(value):
        if value not in {"native", "browser"}:
            raise ValueError("unknown acquisition strategy: " + str(value))
        return value

    @staticmethod
    def attempts_for_strategy(row, strategy=None):
        strategy = AcquisitionFrontier._strategy(strategy or row.get("active_strategy") or "native")
        counts = row.get("strategy_attempts") or {}
        return counts.get(strategy, 0) + counts.get("legacy_unknown", 0)

    @staticmethod
    def _retryable(row, strategy):
        if row is None or row["status"] != "failed":
            return False
        return (row["attempts"] if strategy is None else AcquisitionFrontier.attempts_for_strategy(row, strategy)) < 2

    def enqueue(self, *, target_id, url, source_id, priority=0, payload=None):
        if not target_id or not url or not source_id:
            raise ValueError("target_id, url and source_id are required")
        priority = finite_priority(priority)
        encoded = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM acquisition_frontier WHERE target_id=?", (target_id,)).fetchone()
            if row is None:
                db.execute("""INSERT INTO acquisition_frontier
                    (target_id,url,source_id,priority,payload,updated) VALUES (?,?,?,?,?,?)""",
                    (target_id, url, source_id, priority, encoded, time.time()))
            elif row["status"] == "pending":
                # Repeated discovery may improve priority/provenance but never
                # reopens a completed or blocked target.
                merged = {**json.loads(row["payload"]), **(payload or {})}
                db.execute("UPDATE acquisition_frontier SET priority=?,payload=?,updated=? WHERE target_id=?",
                           (max(priority, row["priority"]), json.dumps(merged, ensure_ascii=False, sort_keys=True),
                            time.time(), target_id))
            return self._row(db.execute("SELECT * FROM acquisition_frontier WHERE target_id=?", (target_id,)).fetchone())

    def claim_next(self, *, default_strategy="native"):
        self._strategy(default_strategy)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT * FROM acquisition_frontier WHERE status='pending'
                                ORDER BY priority DESC, position ASC LIMIT 1""").fetchone()
            if row is None:
                return None
            payload = json.loads(row["payload"])
            strategy = self._strategy((payload.get("entry") or {}).get("acquisition_strategy")
                                      or row["active_strategy"] or default_strategy)
            db.execute("UPDATE acquisition_frontier SET status='running',active_strategy=?,updated=? WHERE target_id=?",
                       (strategy, time.time(), row["target_id"]))
            return self._row(db.execute("SELECT * FROM acquisition_frontier WHERE target_id=?", (row["target_id"],)).fetchone())

    def finish(self, target_id, *, status, reason=None, operation_started=False, result_ref=None, strategy=None):
        if result_ref is not None:
            if (not isinstance(result_ref, str) or not result_ref or PureWindowsPath(result_ref).drive
                    or PureWindowsPath(result_ref).is_absolute() or PurePosixPath(result_ref).is_absolute()
                    or ".." in PurePosixPath(result_ref.replace(chr(92), "/")).parts):
                raise ValueError("result_ref must stay relative to the session")
        if status not in {"completed", "failed", "budget_deferred", "blocked"}:
            raise ValueError("invalid terminal/deferred acquisition status")
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM acquisition_frontier WHERE target_id=?", (target_id,)).fetchone()
            if row is None or row["status"] != "running":
                raise ValueError("only a claimed acquisition target can finish")
            revision = row["budget_revision"]
            if status == "budget_deferred" and db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='budget_amendments'").fetchone():
                revision = int(db.execute("SELECT COALESCE(MAX(revision),0) FROM budget_amendments").fetchone()[0])
            strategy = self._strategy(strategy or row["active_strategy"] or "native")
            counts = json.loads(row["strategy_attempts"])
            if operation_started:
                counts[strategy] = counts.get(strategy, 0) + 1
            db.execute("""UPDATE acquisition_frontier SET status=?,reason=?,attempts=attempts+?,updated=?,result_ref=?,budget_revision=?,
                          active_strategy=?,strategy_attempts=? WHERE target_id=?""",
                       (status, reason, int(bool(operation_started)), time.time(), result_ref, revision,
                        strategy, json.dumps(counts, sort_keys=True), target_id))
            return self._row(db.execute("SELECT * FROM acquisition_frontier WHERE target_id=?", (target_id,)).fetchone())

    def can_retry(self, target_id, *, strategy=None):
        if strategy is not None:
            self._strategy(strategy)
        with self._db() as db:
            row = self._row(db.execute("SELECT * FROM acquisition_frontier WHERE target_id=?", (target_id,)).fetchone())
            return self._retryable(row, strategy)

    def retry(self, target_id, *, strategy=None):
        """Explicit strategy caps retain cumulative attempts; legacy callers keep total cap."""
        if strategy is not None:
            self._strategy(strategy)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._row(db.execute("SELECT * FROM acquisition_frontier WHERE target_id=?", (target_id,)).fetchone())
            if not self._retryable(row, strategy):
                return False
            payload = row["payload"]
            if strategy is not None:
                payload["entry"] = {**(payload.get("entry") or {}), "acquisition_strategy": strategy}
            db.execute("""UPDATE acquisition_frontier SET status='pending',active_strategy=?,payload=?,updated=?
                          WHERE target_id=?""", (strategy or row["active_strategy"],
                          json.dumps(payload, ensure_ascii=False, sort_keys=True), time.time(), target_id))
            return True

    def reprioritize(self, priorities):
        validated = {key: finite_priority(value) for key, value in priorities.items()}
        changed = []
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            for target_id, priority in validated.items():
                result = db.execute("""UPDATE acquisition_frontier SET priority=?,updated=?
                    WHERE target_id=? AND status='pending'""", (priority, time.time(), target_id))
                if result.rowcount:
                    changed.append(target_id)
        return changed

    def resume_budget_deferred(self, *, budget_revision):
        nonnegative_integer(budget_revision, "budget_revision")
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("""SELECT target_id FROM acquisition_frontier
                WHERE status='budget_deferred' AND budget_revision<? ORDER BY position""", (budget_revision,)).fetchall()
            db.execute("""UPDATE acquisition_frontier SET status='pending',budget_revision=?,updated=?
                WHERE status='budget_deferred' AND budget_revision<?""", (budget_revision, time.time(), budget_revision))
            return [row["target_id"] for row in rows]

    def reconcile_running(self, outcomes):
        """RunContext supplies durable transport outcomes after ledger recovery."""
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            for target_id, outcome in outcomes.items():
                if outcome["status"] not in {"pending", "failed"}:
                    raise ValueError("invalid reconciled acquisition status")
                row = self._row(db.execute("SELECT * FROM acquisition_frontier WHERE target_id=? AND status='running'", (target_id,)).fetchone())
                if row is None:
                    continue
                counts = row["strategy_attempts"]
                strategy = self._strategy(row["active_strategy"] or "native")
                if outcome["operation_started"]:
                    counts[strategy] = counts.get(strategy, 0) + 1
                db.execute("""UPDATE acquisition_frontier SET status=?,reason=?,attempts=attempts+?,updated=?,
                              active_strategy=?,strategy_attempts=? WHERE target_id=? AND status='running'""",
                    (outcome["status"], outcome.get("reason"), int(outcome["operation_started"]), time.time(),
                     strategy, json.dumps(counts, sort_keys=True), target_id))

    def snapshot(self):
        with self._db() as db:
            rows = [self._row(row) for row in db.execute("SELECT * FROM acquisition_frontier ORDER BY position")]
        counts = {}
        for row in rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        return {"items": rows, "counts": counts}
