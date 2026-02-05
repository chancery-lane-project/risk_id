"""
Task management system using SQLite for persistence.

Handles asynchronous task processing for long-running ML operations.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional, Dict, Any

# Database path
DB_PATH = Path(__file__).parent / "tasks.db"
db_lock = Lock()


def init_db():
    """Initialize the tasks database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                result_json TEXT,
                error TEXT,
                progress INTEGER DEFAULT 0
            )
        """)
        # Create index for faster lookups
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON tasks(created_at)")
        conn.commit()


def create_task() -> str:
    """Create a new task and return its ID."""
    task_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    with db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO tasks (task_id, status, created_at, updated_at, progress) VALUES (?, ?, ?, ?, ?)",
                (task_id, "pending", now, now, 0)
            )
            conn.commit()

    return task_id


def update_task(task_id: str, status: str, result: Optional[Dict[str, Any]] = None,
                error: Optional[str] = None, progress: Optional[int] = None):
    """Update task status and optionally store result or error."""
    now = datetime.utcnow().isoformat()
    result_json = json.dumps(result) if result else None

    with db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            if progress is not None:
                conn.execute(
                    """UPDATE tasks
                       SET status = ?, updated_at = ?, result_json = ?, error = ?, progress = ?
                       WHERE task_id = ?""",
                    (status, now, result_json, error, progress, task_id)
                )
            else:
                conn.execute(
                    """UPDATE tasks
                       SET status = ?, updated_at = ?, result_json = ?, error = ?
                       WHERE task_id = ?""",
                    (status, now, result_json, error, task_id)
                )
            conn.commit()


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status and result."""
    with db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (task_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            task = dict(row)
            # Parse result JSON if present
            if task['result_json']:
                task['result'] = json.loads(task['result_json'])
            else:
                task['result'] = None
            del task['result_json']

            return task


def cleanup_old_tasks(days: int = 7):
    """Delete tasks older than specified days."""
    cutoff = datetime.utcnow().timestamp() - (days * 24 * 60 * 60)
    cutoff_iso = datetime.fromtimestamp(cutoff).isoformat()

    with db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "DELETE FROM tasks WHERE created_at < ?",
                (cutoff_iso,)
            )
            conn.commit()
