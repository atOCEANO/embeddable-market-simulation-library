//! Unit newtypes over `f64`. Zero-cost `#[repr(transparent)]` wrappers that keep
//! prices, quantities, notionals, and basis points from being mixed up in
//! signatures and accounting math.

use std::ops::{Add, Div, Mul, Neg, Sub};

macro_rules! scalar_newtype {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(pub f64);

        impl $name {
            /// The wrapped value.
            #[inline]
            pub const fn get(self) -> f64 {
                self.0
            }
        }

        impl Add for $name {
            type Output = $name;

            #[inline]
            fn add(self, rhs: $name) -> $name {
                $name(self.0 + rhs.0)
            }
        }

        impl Sub for $name {
            type Output = $name;

            #[inline]
            fn sub(self, rhs: $name) -> $name {
                $name(self.0 - rhs.0)
            }
        }

        impl Neg for $name {
            type Output = $name;

            #[inline]
            fn neg(self) -> $name {
                $name(-self.0)
            }
        }

        impl Mul<f64> for $name {
            type Output = $name;

            #[inline]
            fn mul(self, rhs: f64) -> $name {
                $name(self.0 * rhs)
            }
        }

        impl Div<f64> for $name {
            type Output = $name;

            #[inline]
            fn div(self, rhs: f64) -> $name {
                $name(self.0 / rhs)
            }
        }
    };
}

scalar_newtype! {
    /// A price level, in quote per base.
    Price
}

scalar_newtype! {
    /// A signed base-asset quantity. Negative is short.
    Qty
}

scalar_newtype! {
    /// A notional value in the quote asset (price times quantity).
    Notional
}

scalar_newtype! {
    /// Basis points: one hundredth of a percent, so 1 bps is 0.0001.
    Bps
}

impl Price {
    /// The notional of `qty` units at this price, in the quote asset. Carries the
    /// sign of `qty`, so a short quantity yields a negative notional.
    #[inline]
    pub fn notional(self, qty: Qty) -> Notional {
        Notional(self.0 * qty.0)
    }
}

impl Bps {
    /// This rate as a plain fraction (`bps / 10_000`).
    #[inline]
    pub fn as_fraction(self) -> f64 {
        self.0 / 10_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::{Bps, Notional, Price, Qty};

    #[test]
    fn price_times_qty_is_notional() {
        assert_eq!(Price(100.0).notional(Qty(2.5)), Notional(250.0));
    }

    #[test]
    fn short_qty_gives_negative_notional() {
        assert_eq!(Price(50.0).notional(Qty(-3.0)), Notional(-150.0));
    }

    #[test]
    fn bps_converts_to_fraction() {
        assert_eq!(Bps(2.0).as_fraction(), 0.0002);
        assert_eq!(Bps(10_000.0).as_fraction(), 1.0);
    }

    #[test]
    fn add_sub_neg() {
        assert_eq!(Qty(2.0) + Qty(3.0), Qty(5.0));
        assert_eq!(Qty(2.0) - Qty(5.0), Qty(-3.0));
        assert_eq!(-Qty(4.0), Qty(-4.0));
    }

    #[test]
    fn scalar_mul_div() {
        assert_eq!(Price(100.0) * 1.5, Price(150.0));
        assert_eq!(Notional(200.0) / 4.0, Notional(50.0));
    }
}
