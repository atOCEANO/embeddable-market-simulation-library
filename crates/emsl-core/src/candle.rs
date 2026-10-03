//! The per-bar OHLCV record. The bar engine owns the time series and the
//! zero-copy windows; this is just one bar.

/// One OHLCV bar. Plain `f64` fields, so a series maps directly onto a numpy
/// `(T, 5)` array with no per-field conversion.
///
/// `repr(C)` fixes the layout to the five fields in declaration order with no
/// padding (five 8-byte floats, 40 bytes, 8-byte aligned). That is what lets the
/// Python layer reinterpret a `&[Candle]` window as a contiguous `(T, 5)` float
/// buffer and hand it back as a zero-copy numpy view (ADR 0008). The layout test
/// below guards that contract.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candle {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    /// Traded volume in BASE units, the same units an order's size is in. The
    /// volume cap compares one against the other and the impact term divides one
    /// by the other, so both are meaningless if this arrives denominated in quote:
    /// the cap stops binding and impact goes to zero, silently and in the caller's
    /// favour (ADRs 0005, 0013).
    pub volume: f64,
}

/// The bars as one row-major run of floats, five to a bar in field order, with no
/// copy. Every zero-copy view the Python layer hands out is built on this, so it is
/// the one place the library reinterprets memory (ADR 0008).
pub fn floats(bars: &[Candle]) -> &[f64] {
    // SAFETY: Candle is repr(C) with five f64 fields and no padding (the layout test
    // below asserts it), so `bars` covers exactly `bars.len() * 5` f64s, aligned as
    // f64 and borrowed for as long as `bars` is.
    unsafe { std::slice::from_raw_parts(bars.as_ptr() as *const f64, bars.len() * 5) }
}

#[cfg(test)]
mod tests {
    use super::{floats, Candle};
    use std::mem::{align_of, size_of};

    #[test]
    fn layout_is_five_packed_floats() {
        assert_eq!(size_of::<Candle>(), 5 * size_of::<f64>());
        assert_eq!(align_of::<Candle>(), align_of::<f64>());
    }

    #[test]
    fn a_slice_reinterprets_as_row_major_floats() {
        let bars = [
            Candle {
                open: 1.0,
                high: 2.0,
                low: 3.0,
                close: 4.0,
                volume: 5.0,
            },
            Candle {
                open: 6.0,
                high: 7.0,
                low: 8.0,
                close: 9.0,
                volume: 10.0,
            },
        ];
        assert_eq!(
            floats(&bars),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        );
    }

    #[test]
    fn an_empty_slice_reinterprets_as_no_floats() {
        // an empty slice's pointer is dangling, which from_raw_parts allows only at
        // length zero
        assert!(floats(&[]).is_empty());
    }
}
