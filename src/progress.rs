use std::fmt::Write as FmtWrite;
use std::io::Read;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use indicatif::{MultiProgress, ProgressBar, ProgressState, ProgressStyle};

/// Shared progress state for the scan pipeline.
///
/// Visual layout while running:
///   Layer  1/38  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  done   42.1 MiB
///   Layer  3/38  ━━━━━━━━━━━╸────────────────────────  28%   7.1 GiB   23 MiB/s
///   Layer  4/38  ━━━━━━━━━━━━━━━━━━╸─────────────────  45%   892 MiB   41 MiB/s
///                ━━━━━━━━━━━━━━━━━━━━╸────────────────  52%   total
///   ⠋ 4281 binaries extracted, 312 with CUDA fatbins
#[derive(Clone)]
pub struct ScanProgress {
    multi: Arc<MultiProgress>,
    total: ProgressBar,
    summary: ProgressBar,
    num_layers: usize,
    visible: bool,
    extracted: Arc<AtomicU64>,
    scanned: Arc<AtomicU64>,
    cache_hits: Arc<AtomicU64>,
}

impl ScanProgress {
    pub fn new(total_compressed_bytes: u64, num_layers: usize, visible: bool) -> Self {
        if !visible {
            return Self {
                multi: Arc::new(MultiProgress::with_draw_target(
                    indicatif::ProgressDrawTarget::hidden(),
                )),
                total: ProgressBar::hidden(),
                summary: ProgressBar::hidden(),
                num_layers,
                visible: false,
                extracted: Arc::new(AtomicU64::new(0)),
                scanned: Arc::new(AtomicU64::new(0)),
                cache_hits: Arc::new(AtomicU64::new(0)),
            };
        }

        let multi = MultiProgress::new();

        // Total bar lives below all per-layer bars (inserted first, layers insert_before it)
        let total = multi.add(ProgressBar::new(total_compressed_bytes));
        total.set_style(
            ProgressStyle::default_bar()
                .template("  {bar:40.white/dim} {percent:>3}%  total  ({bytes_per_sec}, {linear_eta} remaining)")
                .expect("valid template")
                .with_key("linear_eta", format_linear_eta)
                .progress_chars("━╸─"),
        );

        // Summary spinner at the very bottom
        let summary = multi.add(ProgressBar::new_spinner());
        summary.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .expect("valid template"),
        );
        summary.set_message("scanning...");

        Self {
            multi: Arc::new(multi),
            total,
            summary,
            num_layers,
            visible,
            extracted: Arc::new(AtomicU64::new(0)),
            scanned: Arc::new(AtomicU64::new(0)),
            cache_hits: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Create a per-layer progress bar, inserted above the total bar.
    pub fn add_layer(&self, layer_idx: usize, layer_size: u64) -> LayerProgress {
        if !self.visible {
            return LayerProgress {
                bar: ProgressBar::hidden(),
                total: self.total.clone(),
            };
        }

        let width = digit_width(self.num_layers);
        let bar = self
            .multi
            .insert_before(&self.total, ProgressBar::new(layer_size));
        bar.set_style(
            ProgressStyle::default_bar()
                .template(&format!(
                    "  Layer {{prefix:>{width}}}/{}  {{bar:34.cyan/blue}} {{percent:>3}}%  {{total_bytes:<10}}  {{bytes_per_sec}}  {{linear_eta}}",
                    self.num_layers,
                ))
                .expect("valid template")
                .with_key("linear_eta", format_linear_eta)
                .progress_chars("━╸─"),
        );
        bar.set_prefix(format!("{}", layer_idx + 1));

        LayerProgress {
            bar,
            total: self.total.clone(),
        }
    }

    /// Increment the extracted binary count and refresh the summary line.
    pub fn inc_extracted(&self) {
        self.extracted.fetch_add(1, Ordering::Relaxed);
        self.refresh_summary();
    }

    /// Increment the scanned (fatbin-containing) binary count and refresh the summary line.
    pub fn inc_scanned(&self) {
        self.scanned.fetch_add(1, Ordering::Relaxed);
        self.refresh_summary();
    }

    /// Increment the parse cache hit counter and refresh the summary line.
    pub fn inc_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
        self.refresh_summary();
    }

    fn refresh_summary(&self) {
        let extracted = self.extracted.load(Ordering::Relaxed);
        let scanned = self.scanned.load(Ordering::Relaxed);
        let hits = self.cache_hits.load(Ordering::Relaxed);
        if hits > 0 {
            self.summary.set_message(format!(
                "{extracted} binaries extracted, {scanned} scanned ({hits} cached)"
            ));
        } else {
            self.summary
                .set_message(format!("{extracted} binaries extracted, {scanned} scanned"));
        }
    }

    /// Record a layer fully served from the manifest + parse cache.
    ///
    /// Advances the total progress bar by the layer's compressed size (so the
    /// percentage stays accurate) and counts the binaries as extracted, scanned,
    /// and cache-hit in one shot.
    pub fn skip_layer(&self, layer_size: u64, num_binaries: u64) {
        self.total.inc(layer_size);
        self.extracted.fetch_add(num_binaries, Ordering::Relaxed);
        self.scanned.fetch_add(num_binaries, Ordering::Relaxed);
        self.cache_hits.fetch_add(num_binaries, Ordering::Relaxed);
        self.refresh_summary();
    }

    /// Return cache hit/miss counts for post-scan logging.
    pub fn cache_stats(&self) -> (u64, u64) {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let scanned = self.scanned.load(Ordering::Relaxed);
        (hits, scanned.saturating_sub(hits))
    }

    /// Mark all bars as finished and clear them from the terminal.
    pub fn finish(&self) {
        self.total.finish_and_clear();
        self.summary.finish_and_clear();
    }
}

/// Progress tracking for a single layer download.
///
/// Wrapping a reader with `ProgressReader::new(reader, layer_progress)` will
/// drive both the per-layer bar and the aggregate total bar.
#[derive(Clone)]
pub struct LayerProgress {
    bar: ProgressBar,
    total: ProgressBar,
}

impl LayerProgress {
    /// Increment both per-layer and total progress.
    pub fn inc(&self, n: u64) {
        self.bar.inc(n);
        self.total.inc(n);
    }

    /// Remove this layer's bar from the display after it's done.
    pub fn clear(&self) {
        self.bar.finish_and_clear();
    }
}

/// A `Read` wrapper that increments a `LayerProgress` as bytes flow through.
pub struct ProgressReader<R> {
    inner: R,
    layer: LayerProgress,
}

impl<R> ProgressReader<R> {
    pub fn new(inner: R, layer: LayerProgress) -> Self {
        Self { inner, layer }
    }
}

impl<R: Read> Read for ProgressReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        if n > 0 {
            self.layer.inc(n as u64);
        }
        Ok(n)
    }
}

/// Linear ETA: elapsed * (remaining / done).
///
/// Unlike indicatif's built-in `{eta}` which uses an exponential moving average
/// of the transfer rate, this computes ETA from total elapsed time and total bytes
/// moved. One giant layer downloading fast won't make the estimate optimistic
/// while all the small layers still have work to do.
fn format_linear_eta(state: &ProgressState, w: &mut dyn FmtWrite) {
    let pos = state.pos();
    let len = state.len().unwrap_or(0);
    if pos == 0 || len == 0 {
        let _ = write!(w, "...");
        return;
    }
    let elapsed = state.elapsed();
    let remaining = elapsed.mul_f64((len.saturating_sub(pos)) as f64 / pos as f64);
    let secs = remaining.as_secs();
    if secs < 60 {
        let _ = write!(w, "{secs}s");
    } else if secs < 3600 {
        let _ = write!(w, "{}m {:02}s", secs / 60, secs % 60);
    } else {
        let _ = write!(w, "{}h {:02}m", secs / 3600, (secs % 3600) / 60);
    }
}

fn digit_width(n: usize) -> usize {
    n.checked_ilog10().map_or(1, |d| d as usize + 1)
}

#[cfg(test)]
mod tests {
    use crate::progress::digit_width;

    #[test]
    fn digit_width_values() {
        assert_eq!(digit_width(0), 1);
        assert_eq!(digit_width(1), 1);
        assert_eq!(digit_width(9), 1);
        assert_eq!(digit_width(10), 2);
        assert_eq!(digit_width(38), 2);
        assert_eq!(digit_width(100), 3);
    }
}
