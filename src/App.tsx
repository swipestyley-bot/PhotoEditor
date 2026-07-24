import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
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
  autoExposure: number;
  autoWhiteBalance: number;
  autoContrast: number;
  exposure: number;
  contrast: number;
  highlights: number;
  shadows: number;
  whites: number;
  blacks: number;
  saturation: number;
  vibrance: number;
  temperature: number;
  tint: number;
  sharpening: number;
  straighten: number;
  cropX: number;
  cropY: number;
  cropW: number;
  cropH: number;
  noiseReduction: number;
  clarity: number;
  vignetteAmount: number;
  vignetteMidpoint: number;
  shadowHue: number;
  shadowSat: number;
  highlightHue: number;
  highlightSat: number;
}

const DEFAULT_EDIT: EditParams = {
  autoExposure: 0, autoWhiteBalance: 0, autoContrast: 0,
  exposure: 0, contrast: 0, highlights: 0, shadows: 0, whites: 0, blacks: 0,
  saturation: 0, vibrance: 0, temperature: 0, tint: 0, sharpening: 0, straighten: 0,
  cropX: 0, cropY: 0, cropW: 0, cropH: 0,
  noiseReduction: 0, clarity: 0, vignetteAmount: 0, vignetteMidpoint: 0,
  shadowHue: 0, shadowSat: 0, highlightHue: 0, highlightSat: 0,
};

interface ExifInfo {
  camera: string | null;
  lens: string | null;
  iso: string | null;
  aperture: string | null;
  shutter: string | null;
  focalLength: string | null;
}

type CropRect = { x: number; y: number; w: number; h: number };
const ASPECTS = ["Free", "1:1", "4:5", "16:9", "Original"] as const;
type Aspect = (typeof ASPECTS)[number];

type SliderCfg = { key: keyof EditParams; label: string; min: number; max: number };
const ADJUST_GROUPS: { title: string; icon: string; sliders: SliderCfg[] }[] = [
  { title: "Auto", icon: "✦", sliders: [
    { key: "autoExposure", label: "Auto Exposure", min: 0, max: 100 },
    { key: "autoWhiteBalance", label: "Auto White Bal.", min: 0, max: 100 },
    { key: "autoContrast", label: "Auto Contrast", min: 0, max: 100 },
  ] },
  { title: "Tone", icon: "◐", sliders: [
    { key: "exposure", label: "Exposure", min: -100, max: 100 },
    { key: "contrast", label: "Contrast", min: -100, max: 100 },
    { key: "highlights", label: "Highlights", min: -100, max: 100 },
    { key: "shadows", label: "Shadows", min: -100, max: 100 },
    { key: "whites", label: "Whites", min: -100, max: 100 },
    { key: "blacks", label: "Blacks", min: -100, max: 100 },
  ] },
  { title: "Color", icon: "❖", sliders: [
    { key: "temperature", label: "Temperature", min: -100, max: 100 },
    { key: "tint", label: "Tint", min: -100, max: 100 },
    { key: "vibrance", label: "Vibrance", min: -100, max: 100 },
    { key: "saturation", label: "Saturation", min: -100, max: 100 },
  ] },
  { title: "Split Tone", icon: "◑", sliders: [
    { key: "shadowHue", label: "Shadow Hue", min: 0, max: 360 },
    { key: "shadowSat", label: "Shadow Sat", min: 0, max: 100 },
    { key: "highlightHue", label: "Highlight Hue", min: 0, max: 360 },
    { key: "highlightSat", label: "Highlight Sat", min: 0, max: 100 },
  ] },
  { title: "Detail", icon: "◆", sliders: [
    { key: "sharpening", label: "Sharpening", min: 0, max: 100 },
    { key: "clarity", label: "Clarity", min: -100, max: 100 },
    { key: "noiseReduction", label: "Noise Reduction", min: 0, max: 100 },
  ] },
  { title: "Effects", icon: "◎", sliders: [
    { key: "vignetteAmount", label: "Vignette", min: -100, max: 100 },
    { key: "vignetteMidpoint", label: "Midpoint", min: 0, max: 100 },
  ] },
  { title: "Geometry", icon: "▢", sliders: [
    { key: "straighten", label: "Straighten", min: -100, max: 100 },
  ] },
];

type Decision = "keep" | "reject";
type FilterKind = "all" | "blurry" | "eyes" | "duplicates" | "keepers" | "rejects";

const IMAGE_EXTENSIONS = [
  "jpg", "jpeg", "png", "tif", "tiff", "webp", "bmp",
  "cr2", "cr3", "nef", "arw", "dng", "raf", "rw2", "orf", "pef", "srw",
];

function eyesVerdict(e: EyesInfo | null): string {
  if (!e) return "no face";
  if (e.bothClosed) return "closed";
  return "open";
}

function photoLabels(p: Photo): { text: string; cls: string }[] {
  const out: { text: string; cls: string }[] = [];
  if (p.isBlurry) out.push({ text: "Blurry", cls: "blur" });
  if (p.eyes?.bothClosed) out.push({ text: "Eyes shut", cls: "eyes" });
  if (p.cluster != null) out.push({ text: `Duplicate ${p.cluster}`, cls: "dup" });
  return out;
}

export default function App() {
  const [mode, setMode] = useState<"cull" | "edit">("cull");
  const [report, setReport] = useState<FolderReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  // culling
  const [decisions, setDecisions] = useState<Record<string, Decision>>({});
  const [filter, setFilter] = useState<FilterKind>("all");
  const [focusIdx, setFocusIdx] = useState(0);
  const [loupe, setLoupe] = useState(false);
  const [namingOpen, setNamingOpen] = useState(false);
  const [project, setProject] = useState<string | null>(null);

  // editing (per-photo settings)
  const [settings, setSettings] = useState<Record<string, EditParams>>({});
  const [exportCorrected, setExportCorrected] = useState(true);
  const [editPath, setEditPath] = useState<string | null>(null);
  const [comparing, setComparing] = useState(false);
  const [cropMode, setCropMode] = useState(false);
  const [cropAspect, setCropAspect] = useState<Aspect>("Free");
  const [editPreview, setEditPreview] = useState<{ before: string; after: string } | null>(null);
  const [editRendering, setEditRendering] = useState(false);

  async function runAnalysis(command: string, args: Record<string, unknown>) {
    setLoading(true);
    setError(null);
    setStatus(null);
    setReport(null);
    setDecisions({});
    setFilter("all");
    setFocusIdx(0);
    setLoupe(false);
    setSettings({});
    setComparing(false);
    setCropMode(false);
    setCropAspect("Free");
    setMode("cull");
    setProject(null);
    try {
      const r = await invoke<FolderReport>(command, args);
      setReport(r);
      setStatus(`Loaded ${r.photos.length} photos.`);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function openFolder() {
    const dir = await open({ directory: true, title: "Choose a photo folder" });
    if (!dir || Array.isArray(dir)) return;
    await runAnalysis("analyze_library", { folder: dir });
  }

  async function openFiles() {
    const picked = await open({
      multiple: true,
      title: "Choose photos",
      filters: [{ name: "Images (JPG, PNG, RAW)", extensions: IMAGE_EXTENSIONS }],
    });
    if (!picked) return;
    const paths = Array.isArray(picked) ? picked : [picked];
    if (paths.length) await runAnalysis("analyze_files", { paths });
  }

  const photos = report?.photos ?? [];
  const selects = useMemo(() => photos.filter((p) => decisions[p.path] === "keep"), [photos, decisions]);
  const kept = selects.length;
  const rejected = Object.values(decisions).filter((d) => d === "reject").length;

  const shown = useMemo(
    () =>
      photos.filter((p) => {
        switch (filter) {
          case "blurry": return p.isBlurry;
          case "eyes": return p.eyes?.bothClosed ?? false;
          case "duplicates": return p.cluster != null;
          case "keepers": return decisions[p.path] === "keep";
          case "rejects": return decisions[p.path] === "reject";
          default: return true;
        }
      }),
    [photos, filter, decisions],
  );

  const counts: Record<FilterKind, number> = {
    all: photos.length,
    blurry: photos.filter((p) => p.isBlurry).length,
    eyes: photos.filter((p) => p.eyes?.bothClosed).length,
    duplicates: photos.filter((p) => p.cluster != null).length,
    keepers: kept,
    rejects: rejected,
  };

  const focus = Math.min(focusIdx, Math.max(shown.length - 1, 0));
  const activePhoto: Photo | undefined = shown[focus];
  const editPhoto: Photo | undefined = selects.find((p) => p.path === editPath) ?? selects[0];

  function decide(path: string, d: Decision) {
    setDecisions((prev) => {
      const next = { ...prev };
      if (next[path] === d) delete next[path];
      else next[path] = d;
      return next;
    });
  }

  function pickFilter(k: FilterKind) {
    setFilter(k);
    setFocusIdx(0);
    setLoupe(false);
  }

  function goEdit() {
    if (kept === 0) return;
    if (project) setMode("edit");
    else setNamingOpen(true);
  }

  // clamp focus when the visible set changes
  useEffect(() => {
    setFocusIdx((i) => Math.min(Math.max(i, 0), Math.max(shown.length - 1, 0)));
  }, [shown.length]);

  // scroll focused grid card into view
  useEffect(() => {
    if (loupe) return;
    document.querySelector<HTMLElement>(`[data-focus="${focus}"]`)?.scrollIntoView({ block: "nearest" });
  }, [focus, loupe]);

  // culling keyboard (no inputs on this screen)
  useEffect(() => {
    if (mode !== "cull" || !report || namingOpen) return;
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (shown.length === 0) return;
      const move = (n: number) => setFocusIdx(Math.min(Math.max(n, 0), shown.length - 1));
      const cur = shown[focus];
      switch (e.key) {
        case "ArrowRight": case "ArrowDown": e.preventDefault(); move(focus + 1); break;
        case "ArrowLeft": case "ArrowUp": e.preventDefault(); move(focus - 1); break;
        case "p": case "P": decide(cur.path, "keep"); move(focus + 1); break;
        case "x": case "X": decide(cur.path, "reject"); move(focus + 1); break;
        case "Enter": case "e": case "E": setLoupe(true); break;
        case "g": case "G": setLoupe(false); break;
        case "Escape": setLoupe(false); break;
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, report, namingOpen, shown, focus]);

  // edit-mode: hold Space to temporarily show the original ("hold to compare")
  useEffect(() => {
    if (mode !== "edit") return;
    const down = (e: KeyboardEvent) => { if (e.code === "Space") { e.preventDefault(); setComparing(true); } };
    const up = (e: KeyboardEvent) => { if (e.code === "Space") setComparing(false); };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, [mode]);

  function paramsFor(path: string): EditParams {
    return settings[path] ?? DEFAULT_EDIT;
  }
  function updateSetting(path: string, key: keyof EditParams, value: number) {
    setSettings((prev) => ({ ...prev, [path]: { ...(prev[path] ?? DEFAULT_EDIT), [key]: value } }));
  }
  function resetPhoto(path: string) {
    setSettings((prev) => ({ ...prev, [path]: { ...DEFAULT_EDIT } }));
  }
  function syncToAll(fromPath: string) {
    const cur = paramsFor(fromPath);
    setSettings((prev) => {
      const next = { ...prev };
      for (const s of selects) next[s.path] = { ...cur };
      return next;
    });
    setStatus(`Synced these settings to all ${selects.length} select${selects.length === 1 ? "" : "s"}.`);
  }

  function updateCrop(path: string, r: CropRect) {
    setSettings((prev) => ({
      ...prev,
      [path]: { ...(prev[path] ?? DEFAULT_EDIT), cropX: r.x, cropY: r.y, cropW: r.w, cropH: r.h },
    }));
  }
  function toggleCrop() {
    if (!editPhoto) return;
    if (cropMode) { setCropMode(false); return; }
    const p = settings[editPhoto.path] ?? DEFAULT_EDIT;
    if (!(p.cropW > 0)) updateCrop(editPhoto.path, { x: 0, y: 0, w: 1, h: 1 });
    setCropMode(true);
  }
  function resetCrop() {
    if (editPhoto) updateCrop(editPhoto.path, { x: 0, y: 0, w: 0, h: 0 });
  }

  // Live edit preview (shared by the stage + histogram), debounced for real-time.
  const editParamsKey = editPhoto ? JSON.stringify(paramsFor(editPhoto.path)) : "";
  useEffect(() => {
    if (mode !== "edit" || !editPhoto || editPhoto.error) { setEditPreview(null); return; }
    const path = editPhoto.path;
    const params = paramsFor(path);
    let cancelled = false;
    setEditRendering(true);
    const t = setTimeout(async () => {
      try {
        const pv = await invoke<{ before: string; after: string }>("preview_edit", { path, params, cropPreview: cropMode });
        if (!cancelled) setEditPreview(pv);
      } catch {
        if (!cancelled) setEditPreview(null);
      } finally {
        if (!cancelled) setEditRendering(false);
      }
    }, 60);
    return () => { cancelled = true; clearTimeout(t); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, editPhoto?.path, editParamsKey, cropMode]);

  async function exportSelects() {
    if (!selects.length) return;
    const dest = await open({ directory: true, title: "Choose export folder" });
    if (!dest || Array.isArray(dest)) return;
    setError(null);
    setStatus(null);
    try {
      const items = selects.map((p) => ({ path: p.path, params: paramsFor(p.path) }));
      const res = await invoke<ExportResult>("export_selects", { items, dest, corrected: exportCorrected });
      const kind = exportCorrected ? "edited" : "original";
      const failed = res.errors.length ? ` — ${res.errors.length} failed` : "";
      setStatus(`Exported ${res.copied} ${kind} photo${res.copied === 1 ? "" : "s"} to ${res.dest}${failed}`);
    } catch (e) {
      setError(String(e));
    }
  }

  const folderBase = report?.folder ? report.folder.split(/[\\/]/).filter(Boolean).pop() ?? "Untitled" : "Untitled";
  const libraryNav: { kind: FilterKind; label: string; icon: string }[] = [
    { kind: "all", label: "All photos", icon: "▦" },
    { kind: "blurry", label: "Blurry", icon: "◐" },
    { kind: "eyes", label: "Eyes closed", icon: "◡" },
    { kind: "duplicates", label: "Duplicates", icon: "⧉" },
  ];
  const selectionNav: { kind: FilterKind; label: string; icon: string }[] = [
    { kind: "keepers", label: "Selects", icon: "✓" },
    { kind: "rejects", label: "Rejects", icon: "✕" },
  ];

  const topbar = (
    <header className="topbar">
      <div className="brand"><span className="brand-mark" />Culler</div>
      <nav className="tabs">
        <button className={"tab" + (mode === "cull" ? " active" : "")} onClick={() => setMode("cull")}>Cull</button>
        <button className={"tab" + (mode === "edit" ? " active" : "")} disabled={kept === 0} onClick={goEdit}>
          Edit{kept > 0 ? ` · ${kept}` : ""}
        </button>
      </nav>
      {mode === "edit" && <span className="topbar-project">{project ?? "Untitled"}</span>}
      <div className="spacer" />
      {mode === "cull" && report && (
        <div className="segmented">
          <button className={"seg" + (!loupe ? " active" : "")} onClick={() => setLoupe(false)} title="Grid (G)">▦ Grid</button>
          <button className={"seg" + (loupe ? " active" : "")} disabled={!shown.length} onClick={() => setLoupe(true)} title="Loupe (E)">▢ Loupe</button>
        </div>
      )}
    </header>
  );

  // ============================ EDIT MODE ============================
  if (mode === "edit") {
    return (
      <div className="shell">
        {topbar}
        <div className="edit-body">
          <main className="stage-wrap">
            {editPhoto ? (
              <EditStage
                photo={editPhoto}
                params={paramsFor(editPhoto.path)}
                preview={editPreview}
                rendering={editRendering}
                comparing={comparing}
                onCompare={setComparing}
                cropMode={cropMode}
                cropAspect={cropAspect}
                onAspect={setCropAspect}
                onCropChange={(r) => updateCrop(editPhoto.path, r)}
                onResetCrop={resetCrop}
                onExitCrop={() => setCropMode(false)}
              />
            ) : (
              <div className="placeholder"><div className="big">🎞️</div>No selects to edit.</div>
            )}
          </main>

          <aside className="panel edit-panel">
            <Histogram src={editPreview?.after ?? null} />
            {editPhoto && <MetaInfo path={editPhoto.path} />}
            {editPhoto ? (
              <AdjustPanel
                params={paramsFor(editPhoto.path)}
                onChange={(k, v) => updateSetting(editPhoto.path, k, v)}
                onReset={() => resetPhoto(editPhoto.path)}
                cropMode={cropMode}
                onToggleCrop={toggleCrop}
              />
            ) : (
              <div className="panel-empty">No selects</div>
            )}
          </aside>
        </div>

        <div className="filmstrip-bar">
          {selects.map((p) => (
            <button
              key={p.path}
              className={"film" + (editPhoto?.path === p.path ? " active" : "")}
              onClick={() => setEditPath(p.path)}
              title={p.name}
            >
              <img src={p.thumbnail} alt={p.name} loading="lazy" />
            </button>
          ))}
        </div>

        <footer className="statusbar">
          <button className="ghost sm" onClick={() => setMode("cull")}>← Back to culling</button>
          {status && <span className="status-msg">{status}</span>}
          {error && <span className="status-msg err">{error}</span>}
          <div className="spacer" />
          <button className="primary" onClick={() => editPhoto && syncToAll(editPhoto.path)} disabled={kept === 0} title="Copy this photo's sliders to every select">
            Sync to all selects
          </button>
          <label className="toggle" title="Export corrected JPEGs instead of untouched originals">
            <input type="checkbox" checked={exportCorrected} onChange={(e) => setExportCorrected(e.target.checked)} />
            edited
          </label>
          <button className="confirm" onClick={exportSelects} disabled={kept === 0}>Export ({kept}) →</button>
        </footer>
      </div>
    );
  }

  // ============================ CULL MODE ============================
  return (
    <div className="shell">
      {topbar}
      <div className="workspace">
        <aside className="sidebar">
          <button className="primary block" onClick={openFolder} disabled={loading}>
            {loading ? "Analyzing…" : "Import folder"}
          </button>
          <button className="ghost block" onClick={openFiles} disabled={loading}>Import files</button>

          {report && (
            <>
              <div className="side-head mt">Library</div>
              {libraryNav.map((n) => (
                <button key={n.kind} className={"nav-row" + (filter === n.kind ? " active" : "")} onClick={() => pickFilter(n.kind)}>
                  <span className="nav-ic">{n.icon}</span>
                  <span className="nav-label">{n.label}</span>
                  <span className="nav-count">{counts[n.kind]}</span>
                </button>
              ))}
              <div className="side-head mt">Selection</div>
              {selectionNav.map((n) => (
                <button key={n.kind} className={"nav-row" + (filter === n.kind ? " active" : "")} onClick={() => pickFilter(n.kind)}>
                  <span className={"nav-ic " + n.kind}>{n.icon}</span>
                  <span className="nav-label">{n.label}</span>
                  <span className="nav-count">{counts[n.kind]}</span>
                </button>
              ))}
              <div className="side-foot">
                <div className="prog-row"><span>Reviewed</span><span>{kept + rejected} / {photos.length}</span></div>
                <div className="progbar"><div className="progbar-fill" style={{ width: `${photos.length ? ((kept + rejected) / photos.length) * 100 : 0}%` }} /></div>
              </div>
            </>
          )}
        </aside>

        <main className="stage-wrap">
          {!report && !loading && (
            <div className="placeholder"><div className="big">📷</div>Import a folder or files to start culling.</div>
          )}
          {loading && (
            <div className="placeholder"><div className="spinner" />Analyzing photos…</div>
          )}
          {report && !loading && shown.length === 0 && (
            <div className="placeholder muted">Nothing matches this view.</div>
          )}
          {report && !loading && shown.length > 0 && !loupe && (
            <div className="grid">
              {shown.map((p, i) => {
                const d = decisions[p.path];
                return (
                  <figure
                    key={p.path}
                    data-focus={i}
                    className={"card" + (d ? " " + d : "") + (i === focus ? " focused" : "")}
                    onClick={() => { setFocusIdx(i); setLoupe(true); }}
                  >
                    <div className="thumb-wrap">
                      {p.thumbnail ? <img className="thumb" src={p.thumbnail} alt={p.name} loading="lazy" /> : <div className="thumb broken">no preview</div>}
                      <div className="badges">
                        {photoLabels(p).map((l) => <span key={l.text} className={"badge " + l.cls}>{l.text}</span>)}
                      </div>
                      {d && <span className={"flag " + d}>{d === "keep" ? "✓" : "✕"}</span>}
                    </div>
                  </figure>
                );
              })}
            </div>
          )}
          {report && !loading && shown.length > 0 && loupe && activePhoto && (
            <Loupe
              photo={activePhoto}
              index={focus}
              total={shown.length}
              onPrev={() => setFocusIdx(Math.max(focus - 1, 0))}
              onNext={() => setFocusIdx(Math.min(focus + 1, shown.length - 1))}
            />
          )}
        </main>

        <aside className="panel">
          {activePhoto ? (
            <PhotoPanel
              photo={activePhoto}
              decision={decisions[activePhoto.path]}
              onKeep={() => decide(activePhoto.path, "keep")}
              onReject={() => decide(activePhoto.path, "reject")}
            />
          ) : (
            <div className="panel-empty">No photo selected</div>
          )}
        </aside>
      </div>

      <footer className="statusbar">
        <span className="status-msg">
          {report ? <>{photos.length} photos <span className="sep">·</span> <span className="c-keep">{kept} selects</span> <span className="sep">·</span> <span className="c-reject">{rejected} rejects</span></> : "No photos loaded"}
        </span>
        {error && <span className="status-msg err">{error}</span>}
        <div className="spacer" />
        <span className="kbd-hint">P keep · X reject · ←/→ move · E loupe</span>
        <button className="confirm" disabled={kept === 0} onClick={goEdit}>Confirm Selects ({kept}) →</button>
      </footer>

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

// ---- full-screen loupe (in-center) ----
function Loupe({ photo, index, total, onPrev, onNext }: {
  photo: Photo; index: number; total: number; onPrev: () => void; onNext: () => void;
}) {
  const [img, setImg] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    setImg(null);
    invoke<string>("large_preview", { path: photo.path })
      .then((d) => { if (!cancelled) setImg(d); })
      .catch(() => { if (!cancelled) setImg(photo.thumbnail || null); });
    return () => { cancelled = true; };
  }, [photo.path]);
  const labels = photoLabels(photo);
  return (
    <div className="loupe">
      <button className="loupe-nav prev" onClick={onPrev} title="Previous (←)">‹</button>
      <div className="loupe-img-area">
        {img ? <img key={photo.path} className="loupe-img" src={img} alt={photo.name} /> : <div className="loupe-loading">Loading…</div>}
        <div className="loupe-badges">
          {labels.length ? labels.map((l) => <span key={l.text} className={"badge " + l.cls}>{l.text}</span>) : <span className="badge ok">No flags</span>}
        </div>
        <div className="loupe-counter">{index + 1} / {total}</div>
      </div>
      <button className="loupe-nav next" onClick={onNext} title="Next (→)">›</button>
    </div>
  );
}

// ---- right inspector (cull) ----
function PhotoPanel({ photo, decision, onKeep, onReject }: {
  photo: Photo; decision: Decision | undefined; onKeep: () => void; onReject: () => void;
}) {
  const labels = photoLabels(photo);
  return (
    <div className="panel-body">
      <div className="panel-preview">
        {photo.thumbnail ? <img src={photo.thumbnail} alt={photo.name} /> : <div className="thumb broken">no preview</div>}
      </div>
      <div className="panel-name" title={photo.name}>{photo.name}</div>
      <div className="panel-labels">
        {labels.length ? labels.map((l) => <span key={l.text} className={"badge " + l.cls}>{l.text}</span>) : <span className="badge ok">No flags</span>}
      </div>
      <dl className="meta">
        <div><dt>Dimensions</dt><dd>{photo.width} × {photo.height}</dd></div>
        <div><dt>Sharpness</dt><dd className={photo.isBlurry ? "bad" : "good"}>{photo.isBlurry ? "Blurry" : "Sharp"}</dd></div>
        {photo.eyes ? <div><dt>Eyes</dt><dd>{eyesVerdict(photo.eyes)}</dd></div> : <div><dt>Face</dt><dd>none</dd></div>}
        {photo.cluster != null && <div><dt>Duplicate</dt><dd>group {photo.cluster}</dd></div>}
        {photo.burst != null && <div><dt>Burst</dt><dd>#{photo.burst}</dd></div>}
        <div><dt>Captured</dt><dd>{photo.timestamp ?? "—"}</dd></div>
      </dl>
      <div className="panel-decide">
        <button className={"decide reject" + (decision === "reject" ? " on" : "")} onClick={onReject}>✕ Reject <kbd>X</kbd></button>
        <button className={"decide keep" + (decision === "keep" ? " on" : "")} onClick={onKeep}>✓ Keep <kbd>P</kbd></button>
      </div>
    </div>
  );
}

// ---- right inspector (edit): full adjustment panel ----
function AdjustPanel({ params, onChange, onReset, cropMode, onToggleCrop }: {
  params: EditParams;
  onChange: (key: keyof EditParams, value: number) => void;
  onReset: () => void;
  cropMode: boolean;
  onToggleCrop: () => void;
}) {
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const toggle = (t: string) =>
    setCollapsed((prev) => {
      const n = new Set(prev);
      if (n.has(t)) n.delete(t);
      else n.add(t);
      return n;
    });
  return (
    <div className="panel-body adjust">
      <div className="adjust-head">
        <span className="side-head">Adjustments</span>
        <button className="linklike" onClick={onReset}>reset all</button>
      </div>
      {ADJUST_GROUPS.map((g) => {
        const open = !collapsed.has(g.title);
        return (
          <div key={g.title} className="adjust-group">
            <button className="group-head" onClick={() => toggle(g.title)}>
              <span className="group-ic">{g.icon}</span>
              <span className="group-title">{g.title}</span>
              <span className={"group-chev" + (open ? " open" : "")}>▸</span>
            </button>
            {open && (
              <div className="group-body">
                {g.title === "Geometry" && (
                  <button className={"crop-toggle" + (cropMode ? " active" : "")} onClick={onToggleCrop}>
                    ▢ {cropMode ? "Exit crop" : "Crop"}
                  </button>
                )}
                {g.sliders.map((s) => (
                  <SliderRow
                    key={s.key}
                    label={s.label}
                    value={params[s.key]}
                    min={s.min}
                    max={s.max}
                    onChange={(v) => onChange(s.key, v)}
                    onReset={() => onChange(s.key, 0)}
                  />
                ))}
              </div>
            )}
          </div>
        );
      })}
      <p className="panel-note">
        Double-click a slider to reset it. Settings are <strong>per-photo</strong> —
        <strong> Sync to all selects</strong> copies them to the batch.
      </p>
    </div>
  );
}

// ---- histogram (computed from the live preview) ----
function Histogram({ src }: { src: string | null }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (!src) return;
    const img = new Image();
    img.onload = () => {
      const sw = 220;
      const sh = Math.max(1, Math.round((sw * img.naturalHeight) / img.naturalWidth));
      const off = document.createElement("canvas");
      off.width = sw; off.height = sh;
      const octx = off.getContext("2d");
      if (!octx) return;
      octx.drawImage(img, 0, 0, sw, sh);
      const data = octx.getImageData(0, 0, sw, sh).data;
      const bins = 64;
      const r = new Array(bins).fill(0), g = new Array(bins).fill(0), b = new Array(bins).fill(0);
      for (let i = 0; i < data.length; i += 4) {
        r[(data[i] * bins) >> 8]++;
        g[(data[i + 1] * bins) >> 8]++;
        b[(data[i + 2] * bins) >> 8]++;
      }
      const max = Math.max(1, ...r, ...g, ...b);
      ctx.clearRect(0, 0, W, H);
      const draw = (arr: number[], color: string) => {
        ctx.beginPath();
        ctx.moveTo(0, H);
        for (let i = 0; i < bins; i++) ctx.lineTo((i / (bins - 1)) * W, H - (arr[i] / max) * H);
        ctx.lineTo(W, H);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
      };
      ctx.globalCompositeOperation = "lighter";
      draw(r, "rgba(255,86,86,0.5)");
      draw(g, "rgba(80,215,120,0.5)");
      draw(b, "rgba(96,150,255,0.5)");
      ctx.globalCompositeOperation = "source-over";
    };
    img.src = src;
  }, [src]);
  return (
    <div className="histogram">
      <div className="hist-head"><span className="group-ic">▤</span> Histogram</div>
      <canvas ref={ref} width={272} height={92} className="hist-canvas" />
    </div>
  );
}

// ---- EXIF metadata viewer ----
function MetaInfo({ path }: { path: string }) {
  const [exif, setExif] = useState<ExifInfo | null>(null);
  useEffect(() => {
    let cancelled = false;
    setExif(null);
    invoke<ExifInfo>("read_exif", { path })
      .then((e) => { if (!cancelled) setExif(e); })
      .catch(() => { if (!cancelled) setExif(null); });
    return () => { cancelled = true; };
  }, [path]);
  const rows: [string, string | null][] = exif
    ? [
        ["Camera", exif.camera],
        ["Lens", exif.lens],
        ["ISO", exif.iso],
        ["Aperture", exif.aperture],
        ["Shutter", exif.shutter],
        ["Focal", exif.focalLength],
      ]
    : [];
  const shown = rows.filter(([, v]) => v);
  return (
    <div className="meta-info">
      <div className="hist-head"><span className="group-ic">ⓘ</span> Metadata</div>
      {shown.length ? (
        <dl className="meta compact">
          {shown.map(([k, v]) => (
            <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
          ))}
        </dl>
      ) : (
        <div className="meta-empty">No camera EXIF</div>
      )}
    </div>
  );
}

// ---- edit center stage: edited photo, hold to compare, crop mode ----
function EditStage({ photo, params, preview, rendering, comparing, onCompare, cropMode, cropAspect, onAspect, onCropChange, onResetCrop, onExitCrop }: {
  photo: Photo; params: EditParams;
  preview: { before: string; after: string } | null; rendering: boolean;
  comparing: boolean; onCompare: (v: boolean) => void;
  cropMode: boolean; cropAspect: Aspect; onAspect: (a: Aspect) => void;
  onCropChange: (r: CropRect) => void; onResetCrop: () => void; onExitCrop: () => void;
}) {
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null);
  const src = preview ? (comparing ? preview.before : preview.after) : photo.thumbnail;

  if (cropMode) {
    const dims = nat ?? { w: photo.width || 3, h: photo.height || 4 };
    const R =
      cropAspect === "Free" ? null :
      cropAspect === "1:1" ? 1 :
      cropAspect === "4:5" ? 4 / 5 :
      cropAspect === "16:9" ? 16 / 9 :
      dims.w / dims.h; // Original
    const na = R == null ? null : (R * dims.h) / dims.w;
    const rect: CropRect = params.cropW > 0
      ? { x: params.cropX, y: params.cropY, w: params.cropW, h: params.cropH }
      : { x: 0, y: 0, w: 1, h: 1 };
    return (
      <div className="stage">
        <div className="crop-area">
          {src ? (
            <div className="crop-frame">
              <img
                className="crop-img"
                src={src}
                alt={photo.name}
                onLoad={(e) => { const w = e.currentTarget.naturalWidth, h = e.currentTarget.naturalHeight; if (w && h) setNat({ w, h }); }}
              />
              <CropOverlay rect={rect} na={na} onChange={onCropChange} />
            </div>
          ) : (
            <div className="loupe-loading">Loading…</div>
          )}
          {rendering && <div className="rendering">Rendering…</div>}
        </div>
        <div className="crop-toolbar">
          {ASPECTS.map((a) => (
            <button key={a} className={"aspect" + (cropAspect === a ? " active" : "")} onClick={() => onAspect(a)}>{a}</button>
          ))}
          <div className="spacer" />
          <button className="ghost sm" onClick={onResetCrop}>Reset</button>
          <button className="confirm sm" onClick={onExitCrop}>Done</button>
        </div>
      </div>
    );
  }

  return (
    <div className="stage">
      <div className="stage-img-wrap">
        {src ? <img className="stage-img" src={src} alt={photo.name} /> : <div className="thumb broken large">no preview</div>}
        {comparing && <div className="ba-tag ba-tag-l">Original</div>}
        {rendering && !comparing && <div className="rendering">Rendering…</div>}
      </div>
      <div className="stage-tools">
        <button
          className="compare-btn"
          onMouseDown={() => onCompare(true)}
          onMouseUp={() => onCompare(false)}
          onMouseLeave={() => onCompare(false)}
          onTouchStart={() => onCompare(true)}
          onTouchEnd={() => onCompare(false)}
        >
          Hold to compare <kbd>Space</kbd>
        </button>
        <span className="stage-name" title={photo.name}>{photo.name}</span>
      </div>
    </div>
  );
}

const CROP_HANDLES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"] as const;

function clampN(v: number, lo: number, hi: number) {
  return Math.min(Math.max(v, lo), hi);
}

/** Center-fit a crop rect to a normalized aspect (w/h) ratio. */
function fitAspect(r: CropRect, na: number): CropRect {
  const cx = r.x + r.w / 2, cy = r.y + r.h / 2;
  let w = r.w, h = r.h;
  if (w / h > na) w = h * na;
  else h = w / na;
  const x = clampN(cx - w / 2, 0, 1 - w);
  const y = clampN(cy - h / 2, 0, 1 - h);
  return { x, y, w, h };
}

/** Compute the new crop rect for a drag of `mode` by (dnx,dny) normalized. */
function applyDrag(mode: string, s: CropRect, dnx: number, dny: number, na: number | null): CropRect {
  const MIN = 0.05;
  if (mode === "move") {
    return { x: clampN(s.x + dnx, 0, 1 - s.w), y: clampN(s.y + dny, 0, 1 - s.h), w: s.w, h: s.h };
  }
  const L = mode.includes("w"), R = mode.includes("e"), T = mode.includes("n"), B = mode.includes("s");
  let x1 = s.x, y1 = s.y, x2 = s.x + s.w, y2 = s.y + s.h;
  if (L) x1 = clampN(s.x + dnx, 0, x2 - MIN);
  if (R) x2 = clampN(s.x + s.w + dnx, x1 + MIN, 1);
  if (T) y1 = clampN(s.y + dny, 0, y2 - MIN);
  if (B) y2 = clampN(s.y + s.h + dny, y1 + MIN, 1);

  if (na != null) {
    const horiz = L || R, vert = T || B;
    if (horiz && vert) {
      // corner: width drives height, anchor the fixed vertical edge
      let w = x2 - x1;
      let h = w / na;
      if (T) y1 = y2 - h; else y2 = y1 + h;
      if (y1 < 0) { y1 = 0; h = y2 - y1; w = h * na; if (L) x1 = x2 - w; else x2 = x1 + w; }
      if (y2 > 1) { y2 = 1; h = y2 - y1; w = h * na; if (L) x1 = x2 - w; else x2 = x1 + w; }
      if (x1 < 0) { x1 = 0; w = x2 - x1; h = w / na; if (T) y1 = y2 - h; else y2 = y1 + h; }
      if (x2 > 1) { x2 = 1; w = x2 - x1; h = w / na; if (T) y1 = y2 - h; else y2 = y1 + h; }
    } else if (horiz) {
      // e/w edge: width drives, center vertically
      let w = x2 - x1;
      let h = w / na;
      const cy = s.y + s.h / 2;
      const maxH = Math.min(cy, 1 - cy) * 2;
      if (h > maxH) { h = maxH; w = h * na; if (L) x1 = x2 - w; else x2 = x1 + w; }
      y1 = cy - h / 2; y2 = cy + h / 2;
    } else if (vert) {
      // n/s edge: height drives, center horizontally
      let h = y2 - y1;
      let w = h * na;
      const cx = s.x + s.w / 2;
      const maxW = Math.min(cx, 1 - cx) * 2;
      if (w > maxW) { w = maxW; h = w / na; if (T) y1 = y2 - h; else y2 = y1 + h; }
      x1 = cx - w / 2; x2 = cx + w / 2;
    }
  }
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
}

// ---- interactive crop overlay ----
function CropOverlay({ rect, na, onChange }: { rect: CropRect; na: number | null; onChange: (r: CropRect) => void }) {
  const ref = useRef<HTMLDivElement>(null);

  // Re-fit when the aspect changes.
  useEffect(() => {
    if (na == null) return;
    const f = fitAspect(rect, na);
    if (Math.abs(f.w - rect.w) > 1e-3 || Math.abs(f.h - rect.h) > 1e-3 || Math.abs(f.x - rect.x) > 1e-3 || Math.abs(f.y - rect.y) > 1e-3) {
      onChange(f);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [na]);

  function start(mode: string, e: ReactPointerEvent) {
    e.preventDefault();
    e.stopPropagation();
    const box = ref.current?.getBoundingClientRect();
    if (!box || box.width === 0) return;
    const s = { ...rect };
    const sx = e.clientX, sy = e.clientY;
    const move = (ev: PointerEvent) => {
      onChange(applyDrag(mode, s, (ev.clientX - sx) / box.width, (ev.clientY - sy) / box.height, na));
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  const boxStyle = { left: `${rect.x * 100}%`, top: `${rect.y * 100}%`, width: `${rect.w * 100}%`, height: `${rect.h * 100}%` };
  return (
    <div className="crop-overlay" ref={ref}>
      <div className="crop-box" style={boxStyle} onPointerDown={(e) => start("move", e)}>
        <div className="crop-thirds" />
        {CROP_HANDLES.map((h) => (
          <div key={h} className={"crop-handle h-" + h} onPointerDown={(e) => start(h, e)} />
        ))}
      </div>
    </div>
  );
}

// ---- project prompt ----
function ProjectPrompt({ defaultName, count, onConfirm, onCancel }: {
  defaultName: string; count: number; onConfirm: (name: string) => void; onCancel: () => void;
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
          <button className="confirm" disabled={!name.trim()} onClick={() => onConfirm(name.trim())}>Start editing →</button>
        </div>
      </div>
    </div>
  );
}

function SliderRow({ label, value, min, max, onChange, onReset }: {
  label: string; value: number; min: number; max: number; onChange: (v: number) => void; onReset: () => void;
}) {
  const sign = min < 0 && value > 0 ? "+" : "";
  return (
    <div className="slider-row">
      <div className="slider-top">
        <span className="slider-label">{label}</span>
        <span className={"slider-val" + (value !== 0 ? " set" : "")}>{sign}{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        className={(min < 0 ? "bipolar" : "") + (value !== 0 ? " set" : "")}
        onChange={(e) => onChange(+e.target.value)}
        onDoubleClick={onReset}
        title="Double-click to reset"
      />
    </div>
  );
}
