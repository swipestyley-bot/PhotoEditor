import { useEffect, useMemo, useState, type ReactNode } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import "./App.css";

interface EyesInfo {
  leftEar: number;
  rightEar: number;
  bothClosed: boolean;
  anyClosed: boolean;
}

interface Photo {
  path: string;
  name: string;
  width: number;
  height: number;
  thumbnail: string;
  frameBlur: number;
  subjectBlur: number | null;
  isBlurry: boolean;
  hasFace: boolean;
  eyes: EyesInfo | null;
  phash: string;
  timestamp: string | null;
  cluster: number | null;
  burst: number | null;
  error: string | null;
}

interface FolderReport {
  folder: string;
  photos: Photo[];
  duplicateGroups: number;
  burstGroups: number;
}

interface ExportResult {
  copied: number;
  dest: string;
  errors: string[];
}

type Decision = "keep" | "reject";

type FilterKind =
  | "all"
  | "blurry"
  | "eyes"
  | "duplicates"
  | "keepers"
  | "rejects"
  | "cluster"
  | "burst";

interface Filter {
  kind: FilterKind;
  id?: number;
}

// Accepted extensions for the "Import files…" picker (no leading dots).
const IMAGE_EXTENSIONS = [
  "jpg", "jpeg", "png", "tif", "tiff", "webp", "bmp",
  "cr2", "cr3", "nef", "arw", "dng", "raf", "rw2", "orf", "pef", "srw",
];

function eyesVerdict(e: EyesInfo | null): string {
  if (!e) return "no face";
  if (e.bothClosed) return "both closed";
  if (e.anyClosed) return "one closed";
  return "open";
}

export default function App() {
  const [report, setReport] = useState<FolderReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>({ kind: "all" });
  const [selected, setSelected] = useState<Photo | null>(null);
  const [decisions, setDecisions] = useState<Record<string, Decision>>({});
  const [exportMsg, setExportMsg] = useState<string | null>(null);
  const [focusIdx, setFocusIdx] = useState(0);

  async function runAnalysis(command: string, args: Record<string, unknown>) {
    setLoading(true);
    setError(null);
    setExportMsg(null);
    setReport(null);
    setSelected(null);
    setDecisions({});
    setFilter({ kind: "all" });
    setFocusIdx(0);
    try {
      const r = await invoke<FolderReport>(command, args);
      setReport(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function openFolder() {
    setError(null);
    const dir = await open({ directory: true, title: "Choose a photo folder" });
    if (!dir || Array.isArray(dir)) return;
    await runAnalysis("analyze_library", { folder: dir });
  }

  async function openFiles() {
    setError(null);
    const picked = await open({
      multiple: true,
      title: "Choose photos",
      filters: [{ name: "Images (JPG, PNG, RAW)", extensions: IMAGE_EXTENSIONS }],
    });
    if (!picked) return;
    const paths = Array.isArray(picked) ? picked : [picked];
    if (paths.length === 0) return;
    await runAnalysis("analyze_files", { paths });
  }

  const photos = report?.photos ?? [];

  async function exportKept() {
    const keptPaths = photos
      .filter((p) => decisions[p.path] === "keep")
      .map((p) => p.path);
    if (keptPaths.length === 0) return;
    const dest = await open({ directory: true, title: "Choose export folder" });
    if (!dest || Array.isArray(dest)) return;
    setError(null);
    setExportMsg(null);
    try {
      const res = await invoke<ExportResult>("export_kept", {
        paths: keptPaths,
        dest,
      });
      const failed = res.errors.length
        ? ` — ${res.errors.length} failed (${res.errors[0]})`
        : "";
      setExportMsg(
        `Copied ${res.copied} select${res.copied === 1 ? "" : "s"} to ${res.dest}${failed}`,
      );
    } catch (e) {
      setError(String(e));
    }
  }

  const kept = Object.values(decisions).filter((d) => d === "keep").length;
  const rejected = Object.values(decisions).filter((d) => d === "reject").length;

  const shown = useMemo(
    () =>
      photos.filter((p) => {
        switch (filter.kind) {
          case "blurry":
            return p.isBlurry;
          case "eyes":
            return p.eyes?.anyClosed ?? false;
          case "duplicates":
            return p.cluster != null;
          case "keepers":
            return decisions[p.path] === "keep";
          case "rejects":
            return decisions[p.path] === "reject";
          case "cluster":
            return p.cluster === filter.id;
          case "burst":
            return p.burst === filter.id;
          default:
            return true;
        }
      }),
    [photos, filter, decisions],
  );

  const chips: { kind: FilterKind; label: string; count: number }[] = [
    { kind: "all", label: "All", count: photos.length },
    { kind: "blurry", label: "Blurry", count: photos.filter((p) => p.isBlurry).length },
    { kind: "eyes", label: "Eyes closed", count: photos.filter((p) => p.eyes?.anyClosed).length },
    { kind: "duplicates", label: "Duplicates", count: photos.filter((p) => p.cluster != null).length },
    { kind: "keepers", label: "Selects", count: kept },
    { kind: "rejects", label: "Rejects", count: rejected },
  ];

  function decide(path: string, d: Decision) {
    setDecisions((prev) => {
      const next = { ...prev };
      if (next[path] === d) delete next[path];
      else next[path] = d;
      return next;
    });
  }

  function showGroup(kind: "cluster" | "burst", id: number) {
    setFilter({ kind, id });
    setSelected(null);
    setFocusIdx(0);
  }

  function pickFilter(kind: FilterKind) {
    setFilter({ kind });
    setFocusIdx(0);
  }

  // Keep the focus cursor inside the visible set as filters change.
  useEffect(() => {
    setFocusIdx((i) => Math.min(Math.max(i, 0), Math.max(shown.length - 1, 0)));
  }, [shown.length]);

  // Scroll the focused card into view.
  useEffect(() => {
    document
      .querySelector<HTMLElement>(`[data-focus="${focusIdx}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [focusIdx, shown]);

  // Keyboard shortcuts for fast review: P = keep, X = reject, arrows = move.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (!report || shown.length === 0) return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;

      const idx = Math.min(focusIdx, shown.length - 1);
      const cur = shown[idx];
      const move = (ni: number) => {
        const c = Math.min(Math.max(ni, 0), shown.length - 1);
        setFocusIdx(c);
        if (selected) setSelected(shown[c]);
      };

      switch (e.key) {
        case "ArrowRight":
        case "ArrowDown":
          e.preventDefault();
          move(idx + 1);
          break;
        case "ArrowLeft":
        case "ArrowUp":
          e.preventDefault();
          move(idx - 1);
          break;
        case "p":
        case "P":
          decide(cur.path, "keep");
          move(idx + 1);
          break;
        case "x":
        case "X":
          decide(cur.path, "reject");
          move(idx + 1);
          break;
        case "Enter":
          setSelected(cur);
          break;
        case "Escape":
          setSelected(null);
          break;
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [report, shown, focusIdx, selected]);

  const activeGroupLabel =
    filter.kind === "cluster"
      ? `Duplicate group ${filter.id}`
      : filter.kind === "burst"
        ? `Burst ${filter.id}`
        : null;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          Photo Culling <span className="brand-sub">alpha</span>
        </div>
        <button className="primary" onClick={openFolder} disabled={loading}>
          {loading ? "Analyzing…" : "Import folder…"}
        </button>
        <button className="primary" onClick={openFiles} disabled={loading}>
          Import files…
        </button>
        {report && (
          <button
            className="primary"
            onClick={exportKept}
            disabled={loading || kept === 0}
            title={kept === 0 ? "Mark photos as a Select (P / ✓) first" : undefined}
          >
            Export Selects ({kept})
          </button>
        )}
        {report && (
          <div className="summary">
            <strong>{photos.length}</strong> photos
            <span className="dot">·</span>
            <strong>{kept}</strong> selects
            <span className="dot">·</span>
            <strong>{rejected}</strong> rejects
            <span className="hint">P = keep · X = reject · ←/→ = move</span>
          </div>
        )}
      </header>

      {error && <div className="error">⚠ {error}</div>}
      {exportMsg && <div className="notice">✓ {exportMsg}</div>}

      {report && (
        <nav className="filters">
          {chips.map((c) => (
            <button
              key={c.kind}
              className={"chip" + (filter.kind === c.kind ? " active" : "")}
              onClick={() => pickFilter(c.kind)}
            >
              {c.label} <span className="chip-count">{c.count}</span>
            </button>
          ))}
          {activeGroupLabel && (
            <button className="chip active" onClick={() => pickFilter("all")}>
              {activeGroupLabel} ✕
            </button>
          )}
        </nav>
      )}

      {loading && (
        <div className="placeholder">
          <div className="spinner" />
          Decoding images, detecting faces &amp; finding duplicates…
        </div>
      )}

      {!loading && !report && !error && (
        <div className="placeholder">
          <div className="big">📷</div>
          Import a folder or files to see blur, eyes-closed, and duplicate
          analysis as a grid.
        </div>
      )}

      {report && shown.length === 0 && (
        <div className="placeholder muted">No photos match this filter.</div>
      )}

      <main className="grid">
        {shown.map((p, i) => {
          const d = decisions[p.path];
          return (
            <figure
              key={p.path}
              data-focus={i}
              className={
                "card" + (d ? " " + d : "") + (i === focusIdx ? " focused" : "")
              }
              onClick={() => {
                setFocusIdx(i);
                setSelected(p);
              }}
            >
              <div className="thumb-wrap">
                {p.thumbnail ? (
                  <img className="thumb" src={p.thumbnail} alt={p.name} loading="lazy" />
                ) : (
                  <div className="thumb broken">decode failed</div>
                )}
                <div className="badges">
                  {p.isBlurry && <span className="badge blur">Blurry</span>}
                  {p.eyes?.bothClosed && <span className="badge eyes">Eyes shut</span>}
                  {p.eyes?.anyClosed && !p.eyes?.bothClosed && (
                    <span className="badge eyes1">1 eye</span>
                  )}
                  {p.cluster != null && (
                    <button
                      className="badge dup"
                      onClick={(e) => {
                        e.stopPropagation();
                        showGroup("cluster", p.cluster!);
                      }}
                      title={`Show only duplicate group ${p.cluster}`}
                    >
                      Dup {p.cluster}
                    </button>
                  )}
                </div>
                {d && (
                  <div className={"decision-flag " + d}>
                    {d === "keep" ? "SELECT" : "REJECT"}
                  </div>
                )}
              </div>
              <figcaption className="cap">
                <span className="cap-name" title={p.name}>
                  {p.name}
                </span>
                <span className="actions" onClick={(e) => e.stopPropagation()}>
                  <button
                    className={"act keep" + (d === "keep" ? " on" : "")}
                    onClick={() => decide(p.path, "keep")}
                    title="Keep / Select (P)"
                  >
                    ✓
                  </button>
                  <button
                    className={"act reject" + (d === "reject" ? " on" : "")}
                    onClick={() => decide(p.path, "reject")}
                    title="Reject (X)"
                  >
                    ✕
                  </button>
                </span>
              </figcaption>
            </figure>
          );
        })}
      </main>

      {selected && (
        <DetailPanel
          photo={selected}
          decision={decisions[selected.path]}
          onDecide={decide}
          onShowGroup={showGroup}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}

function DetailPanel({
  photo,
  decision,
  onDecide,
  onShowGroup,
  onClose,
}: {
  photo: Photo;
  decision: Decision | undefined;
  onDecide: (path: string, d: Decision) => void;
  onShowGroup: (kind: "cluster" | "burst", id: number) => void;
  onClose: () => void;
}) {
  return (
    <div className="overlay" onClick={onClose}>
      <div className="detail" onClick={(e) => e.stopPropagation()}>
        <button className="close" onClick={onClose}>
          ✕
        </button>
        {photo.thumbnail ? (
          <img className="detail-img" src={photo.thumbnail} alt={photo.name} />
        ) : (
          <div className="thumb broken large">decode failed</div>
        )}
        <div className="detail-info">
          <h2 title={photo.name}>{photo.name}</h2>
          {photo.error ? (
            <p className="error">{photo.error}</p>
          ) : (
            <table className="meta">
              <tbody>
                <Row label="Dimensions" value={`${photo.width} × ${photo.height}`} />
                <Row
                  label="Sharpness"
                  value={
                    <span className={photo.isBlurry ? "bad" : "good"}>
                      {photo.isBlurry ? "Blurry" : "Sharp"}
                    </span>
                  }
                />
                <Row label="Frame blur" value={photo.frameBlur.toFixed(1)} />
                <Row
                  label="Subject blur"
                  value={
                    photo.subjectBlur != null
                      ? photo.subjectBlur.toFixed(1) + "  (face region, scale-normalized)"
                      : "— (no face)"
                  }
                />
                {photo.eyes ? (
                  <>
                    <Row label="Eyes" value={eyesVerdict(photo.eyes)} />
                    <Row
                      label="EAR (L / R)"
                      value={`${photo.eyes.leftEar.toFixed(3)} / ${photo.eyes.rightEar.toFixed(3)}`}
                    />
                  </>
                ) : (
                  <Row label="Face" value="none detected" />
                )}
                <Row
                  label="Duplicate group"
                  value={
                    photo.cluster != null ? (
                      <button className="linklike" onClick={() => onShowGroup("cluster", photo.cluster!)}>
                        #{photo.cluster} — show group
                      </button>
                    ) : (
                      "—"
                    )
                  }
                />
                <Row
                  label="Burst"
                  value={
                    photo.burst != null ? (
                      <button className="linklike" onClick={() => onShowGroup("burst", photo.burst!)}>
                        #{photo.burst} — show burst
                      </button>
                    ) : (
                      "—"
                    )
                  }
                />
                <Row label="Captured" value={photo.timestamp ?? "no EXIF time"} />
                <Row label="pHash" value={<code>{photo.phash}</code>} />
              </tbody>
            </table>
          )}
          <div className="detail-actions">
            <button
              className={"act keep big" + (decision === "keep" ? " on" : "")}
              onClick={() => onDecide(photo.path, "keep")}
            >
              ✓ Keep
            </button>
            <button
              className={"act reject big" + (decision === "reject" ? " on" : "")}
              onClick={() => onDecide(photo.path, "reject")}
            >
              ✕ Reject
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <tr>
      <th>{label}</th>
      <td>{value}</td>
    </tr>
  );
}
