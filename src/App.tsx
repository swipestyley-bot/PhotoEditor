import { useEffect, useMemo, useState } from "react";
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

interface EditParams {
  exposure: number;
  whiteBalance: number;
  contrast: number;
}

const DEFAULT_EDIT: EditParams = { exposure: 100, whiteBalance: 100, contrast: 100 };

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

const IMAGE_EXTENSIONS = [
  "jpg", "jpeg", "png", "tif", "tiff", "webp", "bmp",
  "cr2", "cr3", "nef", "arw", "dng", "raf", "rw2", "orf", "pef", "srw",
];

/** The AI labels for a single photo (shown in the reviewer & grid). */
function photoLabels(p: Photo): { text: string; cls: string }[] {
  const out: { text: string; cls: string }[] = [];
  if (p.isBlurry) out.push({ text: "Blurry", cls: "blur" });
  if (p.eyes?.bothClosed) out.push({ text: "Eyes shut", cls: "eyes" });
  else if (p.eyes?.anyClosed) out.push({ text: "1 eye closed", cls: "eyes1" });
  if (p.cluster != null) out.push({ text: `Duplicate ${p.cluster}`, cls: "dup" });
  return out;
}

export default function App() {
  const [mode, setMode] = useState<"cull" | "edit">("cull");
  const [report, setReport] = useState<FolderReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [exportMsg, setExportMsg] = useState<string | null>(null);

  // Culling state
  const [decisions, setDecisions] = useState<Record<string, Decision>>({});
  const [filter, setFilter] = useState<Filter>({ kind: "all" });
  const [focusIdx, setFocusIdx] = useState(0);
  const [reviewIdx, setReviewIdx] = useState<number | null>(null);
  const [namingOpen, setNamingOpen] = useState(false);
  const [project, setProject] = useState<string | null>(null);

  // Editing state
  const [editParams, setEditParams] = useState<EditParams>(DEFAULT_EDIT);
  const [editedThumbs, setEditedThumbs] = useState<Record<string, string>>({});
  const [exportCorrected, setExportCorrected] = useState(true);
  const [batchBusy, setBatchBusy] = useState(false);
  const [editSelected, setEditSelected] = useState<Photo | null>(null);

  async function runAnalysis(command: string, args: Record<string, unknown>) {
    setLoading(true);
    setError(null);
    setExportMsg(null);
    setReport(null);
    setDecisions({});
    setFilter({ kind: "all" });
    setFocusIdx(0);
    setReviewIdx(null);
    setEditedThumbs({});
    setMode("cull");
    setProject(null);
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
  const selects = useMemo(
    () => photos.filter((p) => decisions[p.path] === "keep"),
    [photos, decisions],
  );
  const kept = selects.length;
  const rejected = Object.values(decisions).filter((d) => d === "reject").length;

  const shown = useMemo(
    () =>
      photos.filter((p) => {
        switch (filter.kind) {
          case "blurry": return p.isBlurry;
          case "eyes": return p.eyes?.anyClosed ?? false;
          case "duplicates": return p.cluster != null;
          case "keepers": return decisions[p.path] === "keep";
          case "rejects": return decisions[p.path] === "reject";
          case "cluster": return p.cluster === filter.id;
          case "burst": return p.burst === filter.id;
          default: return true;
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

  function pickFilter(kind: FilterKind) {
    setFilter({ kind });
    setFocusIdx(0);
  }

  // Keep focus cursor inside the visible set as filters change.
  useEffect(() => {
    setFocusIdx((i) => Math.min(Math.max(i, 0), Math.max(shown.length - 1, 0)));
  }, [shown.length]);

  useEffect(() => {
    if (reviewIdx != null) return; // grid hidden behind the reviewer
    document
      .querySelector<HTMLElement>(`[data-focus="${focusIdx}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [focusIdx, reviewIdx]);

  // Keyboard shortcuts — CULLING ONLY (edit view has focusable sliders).
  useEffect(() => {
    if (mode !== "cull" || !report || namingOpen) return;
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (shown.length === 0) return;

      const inReview = reviewIdx != null;
      const idx = Math.min(inReview ? reviewIdx! : focusIdx, shown.length - 1);
      const cur = shown[idx];
      const move = (ni: number) => {
        const c = Math.min(Math.max(ni, 0), shown.length - 1);
        if (inReview) setReviewIdx(c);
        else setFocusIdx(c);
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
          if (!inReview) setReviewIdx(idx);
          break;
        case "Escape":
          if (inReview) setReviewIdx(null);
          break;
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, report, namingOpen, shown, focusIdx, reviewIdx]);

  // ---- editing actions ----
  async function batchEditAll() {
    const paths = selects.map((p) => p.path);
    if (paths.length === 0) return;
    setBatchBusy(true);
    setError(null);
    setExportMsg(null);
    try {
      const edits = await invoke<{ path: string; thumbnail: string }[]>("batch_edit", {
        paths,
        params: editParams,
      });
      setEditedThumbs(() => {
        const next: Record<string, string> = {};
        for (const e of edits) next[e.path] = e.thumbnail;
        return next;
      });
      setExportMsg(`Auto-edit applied to ${edits.length} photo${edits.length === 1 ? "" : "s"}.`);
    } catch (e) {
      setError(String(e));
    } finally {
      setBatchBusy(false);
    }
  }

  async function exportSelects() {
    const paths = selects.map((p) => p.path);
    if (paths.length === 0) return;
    const dest = await open({ directory: true, title: "Choose export folder" });
    if (!dest || Array.isArray(dest)) return;
    setError(null);
    setExportMsg(null);
    try {
      const res = await invoke<ExportResult>("export_kept", {
        paths,
        dest,
        edit: exportCorrected ? editParams : null,
      });
      const kind = exportCorrected ? "edited" : "original";
      const failed = res.errors.length ? ` — ${res.errors.length} failed (${res.errors[0]})` : "";
      setExportMsg(`Exported ${res.copied} ${kind} photo${res.copied === 1 ? "" : "s"} to ${res.dest}${failed}`);
    } catch (e) {
      setError(String(e));
    }
  }

  const folderBase =
    report?.folder ? report.folder.split(/[\\/]/).filter(Boolean).pop() ?? "Untitled" : "Untitled";

  // ======================= EDIT VIEW =======================
  if (mode === "edit") {
    return (
      <div className="app">
        <header className="topbar">
          <button className="ghost" onClick={() => { setMode("cull"); setEditSelected(null); }}>
            ← Culling
          </button>
          <div className="brand">
            {project ?? "Project"} <span className="brand-sub">editing</span>
          </div>
          <button className="primary" onClick={batchEditAll} disabled={batchBusy || kept === 0}>
            {batchBusy ? "Editing…" : `Auto-edit all (${kept})`}
          </button>
          <button className="primary" onClick={exportSelects} disabled={kept === 0}>
            Export ({kept})
          </button>
          <label className="toggle" title="Export auto-corrected JPEGs instead of untouched originals">
            <input type="checkbox" checked={exportCorrected} onChange={(e) => setExportCorrected(e.target.checked)} />
            edited
          </label>
          <div className="summary">
            <strong>{kept}</strong> selects in this project
            <span className="hint">adjust sliders in a photo, then Auto-edit all</span>
          </div>
        </header>

        {error && <div className="error">⚠ {error}</div>}
        {exportMsg && <div className="notice">✓ {exportMsg}</div>}

        <main className="grid">
          {selects.map((p) => {
            const thumb = editedThumbs[p.path] ?? p.thumbnail;
            return (
              <figure key={p.path} className="card" onClick={() => setEditSelected(p)}>
                <div className="thumb-wrap">
                  {thumb ? (
                    <img className="thumb" src={thumb} alt={p.name} loading="lazy" />
                  ) : (
                    <div className="thumb broken">decode failed</div>
                  )}
                  <div className="badges">
                    {editedThumbs[p.path] && <span className="badge edited">Edited</span>}
                  </div>
                </div>
                <figcaption className="cap">
                  <span className="cap-name" title={p.name}>{p.name}</span>
                </figcaption>
              </figure>
            );
          })}
        </main>

        {editSelected && (
          <EditDetail
            photo={editSelected}
            editParams={editParams}
            onEditParams={setEditParams}
            onClose={() => setEditSelected(null)}
          />
        )}
      </div>
    );
  }

  // ======================= CULL VIEW =======================
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          Photo Culling <span className="brand-sub">cull</span>
        </div>
        <button className="primary" onClick={openFolder} disabled={loading}>
          {loading ? "Analyzing…" : "Import folder…"}
        </button>
        <button className="primary" onClick={openFiles} disabled={loading}>
          Import files…
        </button>
        {report && shown.length > 0 && (
          <button className="primary" onClick={() => setReviewIdx(Math.min(focusIdx, shown.length - 1))}>
            Review one-by-one
          </button>
        )}
        {report && (
          <button
            className="confirm"
            onClick={() => setNamingOpen(true)}
            disabled={kept === 0}
            title={kept === 0 ? "Keep (P) some photos first" : "Confirm selects and move to editing"}
          >
            Confirm Selects ({kept}) →
          </button>
        )}
        {report && (
          <div className="summary">
            <strong>{photos.length}</strong> photos
            <span className="dot">·</span>
            <strong>{kept}</strong> selects
            <span className="dot">·</span>
            <strong>{rejected}</strong> rejects
            <span className="hint">click a photo to review · P = keep · X = reject · ←/→ = move</span>
          </div>
        )}
      </header>

      {error && <div className="error">⚠ {error}</div>}
      {exportMsg && <div className="notice">✓ {exportMsg}</div>}

      {report && (
        <nav className="filters">
          {chips.map((c) => (
            <button key={c.kind} className={"chip" + (filter.kind === c.kind ? " active" : "")} onClick={() => pickFilter(c.kind)}>
              {c.label} <span className="chip-count">{c.count}</span>
            </button>
          ))}
          {(filter.kind === "cluster" || filter.kind === "burst") && (
            <button className="chip active" onClick={() => pickFilter("all")}>
              {filter.kind === "cluster" ? "Duplicate group" : "Burst"} {filter.id} ✕
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
          Import a folder or files to start culling.
        </div>
      )}

      {report && shown.length === 0 && <div className="placeholder muted">No photos match this filter.</div>}

      <main className="grid">
        {shown.map((p, i) => {
          const d = decisions[p.path];
          return (
            <figure
              key={p.path}
              data-focus={i}
              className={"card" + (d ? " " + d : "") + (i === focusIdx ? " focused" : "")}
              onClick={() => { setFocusIdx(i); setReviewIdx(i); }}
            >
              <div className="thumb-wrap">
                {p.thumbnail ? (
                  <img className="thumb" src={p.thumbnail} alt={p.name} loading="lazy" />
                ) : (
                  <div className="thumb broken">decode failed</div>
                )}
                <div className="badges">
                  {photoLabels(p).map((l) => (
                    <span key={l.text} className={"badge " + l.cls}>{l.text}</span>
                  ))}
                </div>
                {d && <div className={"decision-flag " + d}>{d === "keep" ? "SELECT" : "REJECT"}</div>}
              </div>
              <figcaption className="cap">
                <span className="cap-name" title={p.name}>{p.name}</span>
                <span className="actions" onClick={(e) => e.stopPropagation()}>
                  <button className={"act keep" + (d === "keep" ? " on" : "")} onClick={() => decide(p.path, "keep")} title="Keep (P)">✓</button>
                  <button className={"act reject" + (d === "reject" ? " on" : "")} onClick={() => decide(p.path, "reject")} title="Reject (X)">✕</button>
                </span>
              </figcaption>
            </figure>
          );
        })}
      </main>

      {reviewIdx != null && shown[reviewIdx] && (
        <Reviewer
          photo={shown[reviewIdx]}
          index={reviewIdx}
          total={shown.length}
          decision={decisions[shown[reviewIdx].path]}
          onDecide={(d) => { decide(shown[reviewIdx!].path, d); setReviewIdx((i) => Math.min((i ?? 0) + 1, shown.length - 1)); }}
          onPrev={() => setReviewIdx((i) => Math.max((i ?? 0) - 1, 0))}
          onNext={() => setReviewIdx((i) => Math.min((i ?? 0) + 1, shown.length - 1))}
          onClose={() => setReviewIdx(null)}
        />
      )}

      {namingOpen && (
        <ProjectPrompt
          defaultName={folderBase}
          count={kept}
          onCancel={() => setNamingOpen(false)}
          onConfirm={(name) => { setProject(name); setNamingOpen(false); setMode("edit"); }}
        />
      )}
    </div>
  );
}

// ======================= Full-screen culling reviewer =======================
function Reviewer({
  photo, index, total, decision, onDecide, onPrev, onNext, onClose,
}: {
  photo: Photo;
  index: number;
  total: number;
  decision: Decision | undefined;
  onDecide: (d: Decision) => void;
  onPrev: () => void;
  onNext: () => void;
  onClose: () => void;
}) {
  const [img, setImg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setImg(null);
    setBusy(true);
    invoke<string>("large_preview", { path: photo.path })
      .then((d) => { if (!cancelled) setImg(d); })
      .catch(() => { if (!cancelled) setImg(photo.thumbnail || null); })
      .finally(() => { if (!cancelled) setBusy(false); });
    return () => { cancelled = true; };
  }, [photo.path]);

  const labels = photoLabels(photo);

  return (
    <div className="reviewer">
      <div className="rv-top">
        <span className="rv-count">{index + 1} / {total}</span>
        <span className="rv-name" title={photo.name}>{photo.name}</span>
        <button className="ghost" onClick={onClose}>✕ Back to grid</button>
      </div>

      <div className="rv-stage">
        <button className="rv-nav" onClick={onPrev} title="Previous (←)">‹</button>
        <div className="rv-imgwrap">
          {img ? (
            <img className="rv-img" src={img} alt={photo.name} />
          ) : (
            <div className="rv-loading">{busy ? "Loading…" : "no preview"}</div>
          )}
          <div className="rv-labels">
            {labels.length ? (
              labels.map((l) => <span key={l.text} className={"badge " + l.cls}>{l.text}</span>)
            ) : (
              <span className="badge ok">No flags</span>
            )}
          </div>
          {decision && <div className={"rv-decision " + decision}>{decision === "keep" ? "SELECT" : "REJECT"}</div>}
        </div>
        <button className="rv-nav" onClick={onNext} title="Next (→)">›</button>
      </div>

      <div className="rv-actions">
        <button className={"rv-btn reject" + (decision === "reject" ? " on" : "")} onClick={() => onDecide("reject")}>
          ✕ Reject <kbd>X</kbd>
        </button>
        <button className={"rv-btn keep" + (decision === "keep" ? " on" : "")} onClick={() => onDecide("keep")}>
          ✓ Keep <kbd>P</kbd>
        </button>
      </div>
    </div>
  );
}

// ======================= Project name prompt =======================
function ProjectPrompt({
  defaultName, count, onConfirm, onCancel,
}: {
  defaultName: string;
  count: number;
  onConfirm: (name: string) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState(defaultName);
  return (
    <div className="overlay" onClick={onCancel}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>Name your project</h2>
        <p className="muted">{count} select{count === 1 ? "" : "s"} will move into editing.</p>
        <input
          autoFocus
          className="text-input"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && name.trim()) onConfirm(name.trim()); }}
          placeholder="e.g. Smith Wedding"
        />
        <div className="modal-actions">
          <button className="ghost" onClick={onCancel}>Cancel</button>
          <button className="primary" disabled={!name.trim()} onClick={() => onConfirm(name.trim())}>
            Start editing →
          </button>
        </div>
      </div>
    </div>
  );
}

// ======================= Edit detail (before/after + sliders) =======================
function EditDetail({
  photo, editParams, onEditParams, onClose,
}: {
  photo: Photo;
  editParams: EditParams;
  onEditParams: (p: EditParams) => void;
  onClose: () => void;
}) {
  const [preview, setPreview] = useState<{ before: string; after: string } | null>(null);
  const [rendering, setRendering] = useState(false);
  const [reveal, setReveal] = useState(50);

  useEffect(() => {
    if (photo.error) return;
    let cancelled = false;
    setRendering(true);
    const t = setTimeout(async () => {
      try {
        const pv = await invoke<{ before: string; after: string }>("preview_edit", {
          path: photo.path,
          params: editParams,
        });
        if (!cancelled) setPreview(pv);
      } catch {
        if (!cancelled) setPreview(null);
      } finally {
        if (!cancelled) setRendering(false);
      }
    }, 200);
    return () => { cancelled = true; clearTimeout(t); };
  }, [photo.path, photo.error, editParams.exposure, editParams.whiteBalance, editParams.contrast]);

  return (
    <div className="overlay" onClick={onClose}>
      <div className="detail" onClick={(e) => e.stopPropagation()}>
        <button className="close" onClick={onClose}>✕</button>

        <div className="detail-media">
          {preview ? (
            <div className="ba">
              <img className="ba-img" src={preview.before} alt="before" />
              <img className="ba-img ba-after" src={preview.after} alt="after" style={{ clipPath: `inset(0 0 0 ${reveal}%)` }} />
              <div className="ba-divider" style={{ left: `${reveal}%` }} />
              <div className="ba-tag ba-tag-l">Before</div>
              <div className="ba-tag ba-tag-r">After</div>
              <input className="ba-range" type="range" min={0} max={100} value={reveal} onChange={(e) => setReveal(+e.target.value)} title="Drag to compare" />
            </div>
          ) : photo.thumbnail ? (
            <img className="detail-img" src={photo.thumbnail} alt={photo.name} />
          ) : (
            <div className="thumb broken large">decode failed</div>
          )}
          {rendering && <div className="rendering">Rendering…</div>}
        </div>

        <div className="detail-info">
          <h2 title={photo.name}>{photo.name}</h2>
          <div className="edit-controls">
            <div className="edit-title">Auto-edit strength (applies to the whole project)</div>
            <SliderRow label="Exposure" value={editParams.exposure} onChange={(v) => onEditParams({ ...editParams, exposure: v })} />
            <SliderRow label="White balance" value={editParams.whiteBalance} onChange={(v) => onEditParams({ ...editParams, whiteBalance: v })} />
            <SliderRow label="Contrast" value={editParams.contrast} onChange={(v) => onEditParams({ ...editParams, contrast: v })} />
            <div className="edit-presets">
              <button className="linklike" onClick={() => onEditParams(DEFAULT_EDIT)}>full (100%)</button>
              <button className="linklike" onClick={() => onEditParams({ exposure: 0, whiteBalance: 0, contrast: 0 })}>off (0%)</button>
            </div>
          </div>
          <p className="muted small">
            Drag the divider to compare. These strengths apply to every photo when you click
            <strong> Auto-edit all</strong> and when you <strong>Export</strong> with “edited” on.
          </p>
        </div>
      </div>
    </div>
  );
}

function SliderRow({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <label className="slider-row">
      <span className="slider-label">{label}</span>
      <input type="range" min={0} max={100} value={value} onChange={(e) => onChange(+e.target.value)} />
      <span className="slider-val">{value}%</span>
    </label>
  );
}
