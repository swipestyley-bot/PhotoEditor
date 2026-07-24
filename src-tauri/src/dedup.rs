//! Duplicate / near-duplicate detection for culling.
//!
//! - **Perceptual hashing (pHash)** — median + DCT, robust to resizing/minor
//!   edits, unlike a byte hash.
//! - **Hash distance** — Hamming distance between two hashes or two images.
//! - **Similarity clustering** — single-linkage grouping under a distance
//!   threshold (groups exact + near duplicates).
//! - **EXIF capture time** — for burst grouping (consecutive rapid frames).
//!
//! Same testable-function + synthetic-image-test pattern as `face.rs`. Threshold
//! defaults are marked `VALIDATE:` — tune them against a real photo set.

use std::collections::BTreeMap;
use std::path::Path;

use chrono::{NaiveDateTime, Timelike};
use image::DynamicImage;
use image_hasher::{HashAlg, Hasher, HasherConfig, ImageHash};

/// pHash size per side. 8 => a classic 64-bit pHash.
const PHASH_SIZE: u32 = 8;

/// Default Hamming distance below which two pHashes are treated as
/// near-duplicates.
/// VALIDATE: tune on real photos. For 64-bit pHash, re-encoded/exact copies are
/// typically <= ~4 and visually near-identical frames <= ~10.
pub const DEFAULT_DUP_DISTANCE: u32 = 10;

/// Default max gap between consecutive frames considered part of one burst.
/// VALIDATE: tune to your camera's burst rate (e.g. 0.1s for 10fps bursts).
pub const DEFAULT_BURST_GAP_SECS: i64 = 2;

type PHash = ImageHash<Box<[u8]>>;

fn hasher() -> Hasher {
    // Median + DCT preprocessing == pHash (Krawetz). See image_hasher docs.
    HasherConfig::new()
        .hash_alg(HashAlg::Median)
        .preproc_dct()
        .hash_size(PHASH_SIZE, PHASH_SIZE)
        .to_hasher()
}

/// Perceptual hash (pHash) of an image as a base64 string. Visually similar
/// images produce hashes with a small Hamming distance ([`hash_distance`]).
pub fn perceptual_hash(img: &DynamicImage) -> String {
    hasher().hash_image(img).to_base64()
}

/// Hamming distance between two base64 pHash strings. `0` = identical.
pub fn hash_distance(a: &str, b: &str) -> Result<u32, String> {
    let ha = PHash::from_base64(a).map_err(|e| format!("invalid hash: {e:?}"))?;
    let hb = PHash::from_base64(b).map_err(|e| format!("invalid hash: {e:?}"))?;
    Ok(ha.dist(&hb))
}

/// Hamming distance between two images (hashes both, then compares).
pub fn hash_distance_images(a: &DynamicImage, b: &DynamicImage) -> u32 {
    let h = hasher();
    h.hash_image(a).dist(&h.hash_image(b))
}

/// Cluster hashes by single-linkage: any two hashes within `max_distance` land
/// in the same cluster (so transitive chains of near-dups group together).
/// Returns clusters of indices into `hashes`; singletons included. Clusters and
/// their members are sorted for determinism.
///
/// O(n²) pairwise. VALIDATE: fine for a shoot; for very large libraries switch
/// to a BK-tree / LSH index.
pub fn cluster_by_similarity(
    hashes: &[String],
    max_distance: u32,
) -> Result<Vec<Vec<usize>>, String> {
    let parsed: Vec<PHash> = hashes
        .iter()
        .map(|h| PHash::from_base64(h).map_err(|e| format!("invalid hash: {e:?}")))
        .collect::<Result<_, _>>()?;

    let n = parsed.len();
    let mut uf = UnionFind::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            if parsed[i].dist(&parsed[j]) <= max_distance {
                uf.union(i, j);
            }
        }
    }
    Ok(uf.groups())
}

/// A minimal per-image fingerprint for dedup / burst analysis.
#[derive(Debug, Clone)]
pub struct Fingerprint {
    /// Caller-chosen identifier (e.g. an index into their file list).
    pub id: usize,
    pub phash: String,
    pub timestamp: Option<NaiveDateTime>,
}

/// Build a fingerprint from a file on disk: decode, pHash, and read EXIF time.
pub fn fingerprint(id: usize, path: impl AsRef<Path>) -> Result<Fingerprint, String> {
    let path = path.as_ref();
    let img = crate::vision::decode_image(path)?;
    Ok(Fingerprint {
        id,
        phash: perceptual_hash(&img),
        timestamp: exif_timestamp(path),
    })
}

/// Group fingerprints into bursts by capture time: within a run sorted by
/// timestamp, a frame starts a new burst when its gap to the previous frame
/// exceeds `max_gap_secs`. Frames without a timestamp can't be sequenced, so
/// each is returned as its own singleton burst. Returns lists of `Fingerprint::id`.
pub fn group_bursts(items: &[Fingerprint], max_gap_secs: i64) -> Vec<Vec<usize>> {
    let mut timed: Vec<&Fingerprint> = items.iter().filter(|f| f.timestamp.is_some()).collect();
    timed.sort_by_key(|f| f.timestamp.unwrap());

    let mut bursts: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut last: Option<NaiveDateTime> = None;
    for f in timed {
        let ts = f.timestamp.unwrap();
        let continues = matches!(last, Some(prev) if (ts - prev).num_seconds() <= max_gap_secs);
        if !continues && !current.is_empty() {
            bursts.push(std::mem::take(&mut current));
        }
        current.push(f.id);
        last = Some(ts);
    }
    if !current.is_empty() {
        bursts.push(current);
    }

    for f in items.iter().filter(|f| f.timestamp.is_none()) {
        bursts.push(vec![f.id]);
    }
    bursts
}

/// EXIF capture time (`DateTimeOriginal`), including sub-second precision when
/// the camera recorded it (matters for high-fps bursts). Returns `None` when the
/// file has no readable EXIF timestamp.
///
/// VALIDATE: real cameras occasionally omit sub-second or use odd encodings;
/// confirm against your gear.
pub fn exif_timestamp(path: impl AsRef<Path>) -> Option<NaiveDateTime> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = std::io::BufReader::new(file);
    let exif = exif::Reader::new().read_from_container(&mut reader).ok()?;

    // EXIF DateTimeOriginal is ASCII "YYYY:MM:DD HH:MM:SS".
    let raw = ascii_field(&exif, exif::Tag::DateTimeOriginal)?;
    let mut dt = NaiveDateTime::parse_from_str(raw.trim(), "%Y:%m:%d %H:%M:%S").ok()?;

    if let Some(subsec) = ascii_field(&exif, exif::Tag::SubSecTimeOriginal) {
        if let Ok(frac) = format!("0.{}", subsec.trim()).parse::<f64>() {
            let nanos = (frac * 1_000_000_000.0) as u32;
            dt = dt.with_nanosecond(nanos).unwrap_or(dt);
        }
    }
    Some(dt)
}

fn ascii_field(exif: &exif::Exif, tag: exif::Tag) -> Option<String> {
    let field = exif.get_field(tag, exif::In::PRIMARY)?;
    if let exif::Value::Ascii(ref vecs) = field.value {
        let bytes = vecs.first()?;
        return std::str::from_utf8(bytes)
            .ok()
            .map(|s| s.trim_matches('\0').trim().to_string());
    }
    None
}

// --- union-find (single-linkage clustering) --------------------------------

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, i: usize) -> usize {
        let mut root = i;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        // Path compression.
        let mut cur = i;
        while self.parent[cur] != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }

    fn groups(&mut self) -> Vec<Vec<usize>> {
        let n = self.parent.len();
        let mut by_root: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for i in 0..n {
            let r = self.find(i);
            by_root.entry(r).or_default().push(i);
        }
        let mut clusters: Vec<Vec<usize>> = by_root.into_values().collect();
        for c in &mut clusters {
            c.sort_unstable();
        }
        clusters.sort_by_key(|c| c[0]);
        clusters
    }
}

/// Tauri command: cluster a list of image files into duplicate/near-duplicate
/// groups. Returns clusters of indices into the input `paths`.
#[tauri::command]
pub fn find_duplicate_clusters(
    paths: Vec<String>,
    max_distance: Option<u32>,
) -> Result<Vec<Vec<usize>>, String> {
    let hashes: Vec<String> = paths
        .iter()
        .map(|p| crate::vision::decode_image(Path::new(p)).map(|img| perceptual_hash(&img)))
        .collect::<Result<_, _>>()?;
    cluster_by_similarity(&hashes, max_distance.unwrap_or(DEFAULT_DUP_DISTANCE))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    /// A deterministic, *broadband* grayscale texture: a sum of many sinusoids
    /// weighted 1/k, giving a natural-image-like 1/f spectrum. This is what makes
    /// pHash stable — energy is spread across many DCT coefficients that sit
    /// robustly off the median (unlike a smooth ramp or a single grating, whose
    /// near-zero coefficients flip randomly under a tiny change). `seed` shifts
    /// the phase to produce a clearly different image.
    ///
    /// Calibrated distances (see the removed phash_probe example): a gentle blur
    /// gives ~4–8, a different seed ~26–34 — cleanly straddling
    /// [`DEFAULT_DUP_DISTANCE`] (10).
    fn pattern_image(seed: u32) -> DynamicImage {
        use std::f32::consts::PI;
        let s = seed as f32;
        DynamicImage::ImageRgb8(RgbImage::from_fn(128, 128, |x, y| {
            let fx = x as f32 / 128.0;
            let fy = y as f32 / 128.0;
            let mut v = 0.0f32;
            for k in 1..=6 {
                let kf = k as f32;
                v += (2.0 * PI * kf * fx + s).sin() / kf;
                v += (2.0 * PI * kf * fy + s * 0.7).cos() / kf;
            }
            let lum = (128.0 + 40.0 * v).clamp(0.0, 255.0) as u8;
            Rgb([lum, lum, lum])
        }))
    }

    #[test]
    fn phash_is_deterministic() {
        let img = pattern_image(0);
        assert_eq!(perceptual_hash(&img), perceptual_hash(&img));
        assert_eq!(hash_distance(&perceptual_hash(&img), &perceptual_hash(&img)).unwrap(), 0);
    }

    #[test]
    fn near_duplicate_has_small_distance() {
        // A slightly blurred copy is a near-duplicate -> small pHash distance.
        let original = pattern_image(0);
        let blurred = DynamicImage::ImageRgb8(imageproc::filter::gaussian_blur_f32(
            original.as_rgb8().unwrap(),
            1.0,
        ));
        let d = hash_distance_images(&original, &blurred);
        println!("near-dup distance = {d}");
        assert!(d <= DEFAULT_DUP_DISTANCE, "near-dup distance {d} should be small");
    }

    #[test]
    fn different_images_have_large_distance() {
        let a = pattern_image(0);
        let b = pattern_image(1);
        let d = hash_distance_images(&a, &b);
        println!("different-image distance = {d}");
        assert!(d > DEFAULT_DUP_DISTANCE, "distinct images distance {d} should be large");
    }

    #[test]
    fn clustering_groups_near_dups_and_separates_different() {
        let a = pattern_image(0);
        let a_blur = DynamicImage::ImageRgb8(imageproc::filter::gaussian_blur_f32(
            a.as_rgb8().unwrap(),
            1.0,
        ));
        let b = pattern_image(1);

        // Order: [a, b, a_blur] -> expect {a, a_blur} together, {b} alone.
        let hashes = vec![
            perceptual_hash(&a),
            perceptual_hash(&b),
            perceptual_hash(&a_blur),
        ];
        let clusters = cluster_by_similarity(&hashes, DEFAULT_DUP_DISTANCE).unwrap();
        println!("clusters = {clusters:?}");
        assert!(clusters.contains(&vec![0, 2]), "a and a_blur should cluster: {clusters:?}");
        assert!(clusters.contains(&vec![1]), "b should be its own cluster: {clusters:?}");
    }

    #[test]
    fn bursts_split_on_time_gap() {
        let t = |s: &str| NaiveDateTime::parse_from_str(s, "%Y:%m:%d %H:%M:%S").unwrap();
        let fp = |id, ts: Option<NaiveDateTime>| Fingerprint {
            id,
            phash: String::new(),
            timestamp: ts,
        };
        // Two frames 1s apart, then a 10s gap, then one more, plus one untimed.
        let items = vec![
            fp(0, Some(t("2024:01:15 14:30:00"))),
            fp(1, Some(t("2024:01:15 14:30:01"))),
            fp(2, Some(t("2024:01:15 14:30:11"))),
            fp(3, None),
        ];
        let bursts = group_bursts(&items, DEFAULT_BURST_GAP_SECS);
        println!("bursts = {bursts:?}");
        assert!(bursts.contains(&vec![0, 1]), "0 and 1 are one burst: {bursts:?}");
        assert!(bursts.contains(&vec![2]), "2 starts a new burst: {bursts:?}");
        assert!(bursts.contains(&vec![3]), "untimed frame is a singleton: {bursts:?}");
    }

    #[test]
    fn no_exif_returns_none() {
        // A bare PNG we write has no EXIF -> None (must not panic).
        let dir = std::env::temp_dir();
        let path = dir.join("dedup_no_exif_test.png");
        pattern_image(0).save(&path).unwrap();
        assert!(exif_timestamp(&path).is_none());
        let _ = std::fs::remove_file(&path);
    }
}
