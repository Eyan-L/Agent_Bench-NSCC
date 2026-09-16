import { DatabaseSync } from "node:sqlite";
import { mkdirSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const [dbPath, sessionsDir, readyPath] = process.argv.slice(2);
if (!dbPath || !sessionsDir || !readyPath) {
  throw new Error(
    "usage: sqlite_session_exporter.mjs <database> <sessions-dir> <ready-path>",
  );
}

let stopping = false;
let lastError = "";
process.on("SIGTERM", () => { stopping = true; });
process.on("SIGINT", () => { stopping = true; });

function atomicWrite(path, text) {
  mkdirSync(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${process.pid}`;
  writeFileSync(temporary, text, "utf8");
  renameSync(temporary, path);
}

function safeTranscriptName(sessionId) {
  return /^[A-Za-z0-9._-]+$/.test(sessionId) && ![".", ".."].includes(sessionId)
    ? `${sessionId}.jsonl`
    : null;
}

atomicWrite(readyPath, JSON.stringify({ pid: process.pid, readyAt: Date.now() }));

function exportSnapshot() {
  const db = new DatabaseSync(dbPath, { readOnly: true });
  try {
    db.exec("PRAGMA query_only=ON; PRAGMA busy_timeout=500; BEGIN");
    const tables = new Set(
      db.prepare("SELECT name FROM sqlite_master WHERE type='table'")
        .all()
        .map((row) => String(row.name)),
    );
    for (const required of ["session_nodes", "session_windows", "transcript_events"]) {
      if (!tables.has(required)) return;
    }

    const rows = db.prepare(`
      SELECT n.session_key, n.current_session_id, n.entry_json,
             n.updated_at AS node_updated_at, n.status AS node_status,
             n.parent_session_key, n.spawned_by,
             w.status AS window_status, w.started_at, w.ended_at, w.created_at
        FROM session_nodes AS n
        LEFT JOIN session_windows AS w
          ON w.session_id = n.current_session_id
    `).all();
    const eventQuery = db.prepare(`
      SELECT event_json FROM transcript_events
       WHERE session_id = ? ORDER BY seq
    `);
    const index = {};
    const transcripts = [];
    for (const row of rows) {
      const key = String(row.session_key || "");
      const sessionId = String(row.current_session_id || "");
      if (!key || !sessionId) continue;
      let entry = {};
      try {
        const parsed = JSON.parse(String(row.entry_json || "{}"));
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) entry = parsed;
      } catch {}
      const status = String(
        row.node_status || row.window_status || entry.status || "unknown",
      ).trim().toLowerCase();
      const sessionFile = safeTranscriptName(sessionId);
      entry = {
        ...entry,
        sessionId,
        status,
        updatedAt: row.node_updated_at,
        startedAt: row.started_at,
        endedAt: row.ended_at,
        createdAt: row.created_at,
        ...(sessionFile ? { sessionFile } : {}),
      };
      if (row.spawned_by) entry.spawnedBy = String(row.spawned_by);
      else delete entry.spawnedBy;
      if (row.parent_session_key) entry.parentSessionKey = String(row.parent_session_key);
      else delete entry.parentSessionKey;
      index[key] = entry;
      if (sessionFile) {
        const events = eventQuery.all(sessionId).map((event) => String(event.event_json));
        transcripts.push([sessionFile, events.map((event) => `${event}\n`).join("")]);
      }
    }
    db.exec("COMMIT");

    // Publish transcripts first; sessions.json is the consistency marker read by the watcher.
    mkdirSync(sessionsDir, { recursive: true });
    for (const [name, text] of transcripts) atomicWrite(join(sessionsDir, name), text);
    atomicWrite(join(sessionsDir, "sessions.json"), JSON.stringify(index, null, 2));
    lastError = "";
  } catch (error) {
    try { db.exec("ROLLBACK"); } catch {}
    const message = error instanceof Error ? error.message : String(error);
    if (message !== lastError) {
      console.error(`[agent-bench-session-exporter] ${message}`);
      lastError = message;
    }
  } finally {
    db.close();
  }
}

while (!stopping) {
  try { exportSnapshot(); } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (message !== lastError) {
      console.error(`[agent-bench-session-exporter] ${message}`);
      lastError = message;
    }
  }
  await new Promise((resolve) => setTimeout(resolve, 1000));
}
