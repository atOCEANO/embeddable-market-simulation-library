//! The shared candle series. One immutable copy backs every env through an
//! `Arc`, so thousands of envs share it with no duplication, and a window is a
//! zero-copy borrow into it.

use std::sync::Arc;

use emsl_core::Candle;

/// An immutable OHLCV series shared cheaply across envs.
#[derive(Clone, Debug)]
pub struct Candles {
    bars: Arc<[Candle]>,
}

impl Candles {
    /// Build a series from anything that becomes an `Arc<[Candle]>`, such as a
    /// `Vec<Candle>`. The bars are stored once and shared by clone.
    pub fn new(bars: impl Into<Arc<[Candle]>>) -> Candles {
        Candles { bars: bars.into() }
    }

    /// Number of bars.
    pub fn len(&self) -> usize {
        self.bars.len()
    }

    /// True when the series has no bars.
    pub fn is_empty(&self) -> bool {
        self.bars.is_empty()
    }

    /// The bar at `i`, if it exists.
    pub fn get(&self, i: usize) -> Option<Candle> {
        self.bars.get(i).copied()
    }

    /// The whole series as a slice, a zero-copy borrow.
    pub fn as_slice(&self) -> &[Candle] {
        &self.bars
    }

    /// The `len` bars ending just before `end`, as a zero-copy borrow. Clamped
    /// to what exists: the start saturates at zero and `end` saturates at the
    /// series length, so out-of-range requests return the available bars.
    pub fn window(&self, end: usize, len: usize) -> &[Candle] {
        let end = end.min(self.bars.len());
        let start = end.saturating_sub(len);
        &self.bars[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::Candles;
    use emsl_core::Candle;

    fn series(n: usize) -> Candles {
        let bars: Vec<Candle> = (0..n)
            .map(|i| Candle {
                open: i as f64,
                high: i as f64 + 1.0,
                low: i as f64 - 1.0,
                close: i as f64 + 0.5,
                volume: 100.0,
            })
            .collect();
        Candles::new(bars)
    }

    #[test]
    fn len_and_get() {
        let c = series(5);
        assert_eq!(c.len(), 5);
        assert!(!c.is_empty());
        assert_eq!(c.get(2).unwrap().open, 2.0);
        assert!(c.get(5).is_none());
    }

    #[test]
    fn window_is_the_preceding_bars() {
        let c = series(10);
        let w = c.window(5, 3); // bars 2, 3, 4
        assert_eq!(w.len(), 3);
        assert_eq!(w[0].open, 2.0);
        assert_eq!(w[2].open, 4.0);
    }

    #[test]
    fn window_saturates_at_the_start() {
        let c = series(10);
        let w = c.window(2, 5); // only bars 0, 1 exist before index 2
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].open, 0.0);
    }

    #[test]
    fn window_clamps_past_the_end() {
        let c = series(4);
        let w = c.window(99, 2); // end clamps to 4, so bars 2, 3
        assert_eq!(w.len(), 2);
        assert_eq!(w[1].open, 3.0);
    }

    #[test]
    fn clone_shares_the_same_buffer_zero_copy() {
        let a = series(8);
        let b = a.clone();
        // Same backing allocation: the clone is a refcount bump, not a copy.
        assert!(std::ptr::eq(a.as_slice().as_ptr(), b.as_slice().as_ptr()));
    }
}
