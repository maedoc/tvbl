import fs from "node:fs";
import path from "node:path";

// Per-run JSONL op log: one JSON object per line, append-only. Doubles as a
// replay log — re-executing a run replays these entries deterministically.
export function createOpLog(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const stream = fs.createWriteStream(filePath, { flags: "a" });
  let n = 0;
  return {
    path: filePath,
    append(entry) {
      n += 1;
      stream.write(JSON.stringify({ seq: n, ...entry }) + "\n");
    },
    count: () => n,
    close: () => new Promise((res) => stream.end(res)),
  };
}
