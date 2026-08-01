//! Performance statistics derived from a recorded equity curve and trade log.
//! The conventions are fixed in ADR 0007.

use crate::reporter::Trade;

/// A run's performance summary. Percent fields are in percent (10.0 is 10%);
/// ratios (sharpe, sortino, calmar, profit_factor) are unitless.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stats {
    pub total_return_pct: f64,
    /// Net return after fees as a percent of starting equity. Equity is
    /// fee-inclusive, so this equals `total_return_pct`; it carries the spec's name.
    pub net_profit_pct: f64,
    pub cagr_pct: f64,
    pub sharpe: f64,
    pub sortino: f64,
    pub calmar: f64,
    pub max_drawdown_pct: f64,
    pub volatility_pct: f64,
    /// Fraction of recorded steps spent holding a position, in percent.
    pub exposure_pct: f64,
    /// Fraction of closed trades whose PnL net of the recorded fee is positive
    /// (ADR 0029).
    pub win_rate: f64,
    /// Net profit over net loss across the closed trades, both after the recorded
    /// fee; infinite when nothing lost (ADR 0029).
    pub profit_factor: f64,
    pub num_trades: usize,
    /// Mean net trade PnL as a percent of STARTING equity, not of the equity the
    /// trade opened with, so later trades in a drawn-down account count for their
    /// nominal size rather than their real one (ADR 0007).
    pub avg_trade_pct: f64,
    /// Fills applied over the run. Zero beside orders you know you placed means they
    /// never filled, which a zero-volume series does silently; without this a dead
    /// feed and a strategy that never triggered look identical (ADR 0031).
    pub num_fills: usize,
}

impl Stats {
    /// Compute stats from the starting equity, the per-step equity curve, the
    /// closed trades, and the annualization inputs. Degenerate inputs return
    /// zeros rather than errors, so the result is always defined.
    pub fn compute(
        initial: f64,
        curve: &[f64],
        trades: &[Trade],
        periods_per_year: f64,
        risk_free: f64,
        in_position_steps: usize,
        num_fills: usize,
    ) -> Stats {
        // A non-positive or non-finite annualization has no per-period rate and no
        // square root: `risk_free / periods_per_year` would be NaN and poison sharpe,
        // which is the one thing ADR 0007 says this set must never return. Fall back
        // to no annualization rather than propagating it (ADR 0029).
        let (periods_per_year, risk_free) =
            if periods_per_year.is_finite() && periods_per_year > 0.0 && risk_free.is_finite() {
                (periods_per_year, risk_free)
            } else {
                (1.0, 0.0)
            };

        // The equity path including the starting point.
        let mut series = Vec::with_capacity(curve.len() + 1);
        series.push(initial);
        series.extend_from_slice(curve);

        let final_equity = *series.last().expect("series holds the initial equity");
        let n_returns = series.len() - 1;

        let (total_return, cagr) = if initial > 0.0 && n_returns > 0 {
            let tr = final_equity / initial - 1.0;
            // A wipeout (final equity at or below zero, e.g. a perp bust past zero)
            // annualizes to -100%; computing it as powf of a non-positive ratio
            // would be NaN and poison a sweep's argmax (ADR 0007).
            let cagr = if final_equity > 0.0 {
                (final_equity / initial).powf(periods_per_year / n_returns as f64) - 1.0
            } else {
                -1.0
            };
            (tr, cagr)
        } else {
            (0.0, 0.0)
        };

        // Guard against a non-positive previous equity: after a bust the ratio would
        // flip sign meaninglessly, so those bars contribute a zero return.
        let returns: Vec<f64> = series
            .windows(2)
            .map(|w| if w[0] > 0.0 { w[1] / w[0] - 1.0 } else { 0.0 })
            .collect();

        let rf_per = risk_free / periods_per_year;
        let ann = periods_per_year.sqrt();
        let (sharpe, sortino, volatility) = if returns.len() >= 2 {
            let count = returns.len() as f64;
            let mean = returns.iter().sum::<f64>() / count;
            let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (count - 1.0);
            let std = variance.sqrt();
            let downside = (returns
                .iter()
                .map(|r| (r - rf_per).min(0.0).powi(2))
                .sum::<f64>()
                / count)
                .sqrt();
            let sharpe = if std > 0.0 {
                (mean - rf_per) / std * ann
            } else {
                0.0
            };
            let sortino = if downside > 0.0 {
                (mean - rf_per) / downside * ann
            } else {
                0.0
            };
            (sharpe, sortino, std * ann)
        } else {
            (0.0, 0.0, 0.0)
        };

        let mut peak = series[0];
        let mut max_dd = 0.0_f64;
        for &e in &series {
            if e > peak {
                peak = e;
            }
            if peak > 0.0 {
                // A drawdown is bounded by the peak it falls from: losing everything
                // is 100%. Bad debt takes equity below zero (ADR 0003), and leaving
                // that uncapped reported drawdowns above 100% whose larger divisor
                // gave a DEEPER bust a HIGHER calmar than a shallower one, exactly
                // the argmax poisoning ADR 0007 exists to prevent (ADR 0029).
                max_dd = max_dd.max(((peak - e) / peak).min(1.0));
            }
        }

        let calmar = if max_dd > 0.0 { cagr / max_dd } else { 0.0 };

        let exposure_pct = if !curve.is_empty() {
            in_position_steps as f64 / curve.len() as f64 * 100.0
        } else {
            0.0
        };

        // The trade metrics read `net_pnl`, the whole round trip after fees. On gross
        // PnL a scalper whose edge is smaller than its costs reported a perfect win
        // rate and an infinite profit factor beside a negative total return, the most
        // flattering way a backtest can lie (ADR 0029), and netting only the exit
        // side left half that error in place (ADR 0030). `Trade.pnl` stays gross
        // (ADR 0009), so the log carries both and either can be recomputed.
        let num_trades = trades.len();
        let mut wins = 0usize;
        let mut net_profit = 0.0;
        let mut net_loss = 0.0;
        let mut total_pnl = 0.0;
        for t in trades {
            let net = t.net_pnl;
            total_pnl += net;
            if net > 0.0 {
                wins += 1;
                net_profit += net;
            } else if net < 0.0 {
                net_loss -= net; // accumulate a positive magnitude
            }
        }
        let win_rate = if num_trades > 0 {
            wins as f64 / num_trades as f64
        } else {
            0.0
        };
        let profit_factor = if net_loss > 0.0 {
            net_profit / net_loss
        } else if net_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        let avg_trade_pct = if num_trades > 0 && initial > 0.0 {
            (total_pnl / num_trades as f64) / initial * 100.0
        } else {
            0.0
        };

        Stats {
            total_return_pct: total_return * 100.0,
            net_profit_pct: total_return * 100.0,
            cagr_pct: cagr * 100.0,
            sharpe,
            sortino,
            calmar,
            max_drawdown_pct: max_dd * 100.0,
            volatility_pct: volatility * 100.0,
            exposure_pct,
            win_rate,
            profit_factor,
            num_trades,
            avg_trade_pct,
            num_fills,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Stats;
    use crate::reporter::Trade;
    use emsl_core::Side;

    fn trade(pnl: f64) -> Trade {
        Trade {
            entry_tick: 0,
            exit_tick: 1,
            side: Side::Buy,
            size: 1.0,
            entry_price: 100.0,
            exit_price: 100.0 + pnl,
            fees: 0.0,
            pnl,
            net_pnl: pnl,
            bars_held: 1,
        }
    }

    /// `Stats::compute` with no fills recorded, which is all these cases need.
    fn compute(
        initial: f64,
        curve: &[f64],
        trades: &[Trade],
        periods_per_year: f64,
        risk_free: f64,
        in_position_steps: usize,
    ) -> Stats {
        Stats::compute(
            initial,
            curve,
            trades,
            periods_per_year,
            risk_free,
            in_position_steps,
            0,
        )
    }

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn total_return_and_max_drawdown() {
        // initial 100, curve [120, 90, 110]: peak 120, trough 90 -> 25% drawdown
        let s = compute(100.0, &[120.0, 90.0, 110.0], &[], 252.0, 0.0, 3);
        assert!(approx(s.total_return_pct, 10.0)); // 110 / 100 - 1
        assert!(approx(s.net_profit_pct, 10.0)); // fee-inclusive, equals total return
        assert!(approx(s.max_drawdown_pct, 25.0)); // (120 - 90) / 120
        assert!(approx(s.exposure_pct, 100.0)); // in a position all 3 steps
    }

    #[test]
    fn monotonic_up_has_no_drawdown_and_positive_sharpe() {
        let s = compute(100.0, &[110.0, 120.0, 130.0], &[], 252.0, 0.0, 3);
        assert_eq!(s.max_drawdown_pct, 0.0);
        assert!(s.sharpe > 0.0);
    }

    #[test]
    fn full_scenario_matches_hand_computed_stats() {
        // initial 100, curve [110, 99, 121], ppy 3, rf 0. Returns 0.10, -0.10,
        // 0.222222; mean 0.0740741, sample std 0.1626681, downside dev 0.0577350,
        // all worked out by hand from ADR 0007, independent of the code.
        let trades = [trade(30.0), trade(-10.0), trade(5.0)];
        let s = compute(100.0, &[110.0, 99.0, 121.0], &trades, 3.0, 0.0, 3);
        assert!(approx(s.total_return_pct, 21.0)); // 121/100 - 1
        assert!(approx(s.cagr_pct, 21.0)); // 1.21^(3/3) - 1
        assert!(approx(s.max_drawdown_pct, 10.0)); // (110 - 99) / 110
        assert!((s.sharpe - 0.788679).abs() < 1e-4);
        assert!((s.sortino - 2.222222).abs() < 1e-4);
        assert!(approx(s.calmar, 2.1)); // 0.21 / 0.10
        assert!((s.volatility_pct - 28.175416).abs() < 1e-3);
        assert_eq!(s.num_trades, 3);
        assert!(approx(s.win_rate, 2.0 / 3.0));
        assert!(approx(s.profit_factor, 3.5)); // 35 / 10
        assert!(approx(s.avg_trade_pct, 25.0 / 3.0)); // (25/3) / 100 * 100
    }

    #[test]
    fn trade_stats_win_rate_and_profit_factor() {
        let trades = [trade(100.0), trade(-50.0), trade(25.0)];
        let s = compute(1000.0, &[1000.0], &trades, 252.0, 0.0, 1);
        assert_eq!(s.num_trades, 3);
        assert!(approx(s.win_rate, 2.0 / 3.0));
        assert!(approx(s.profit_factor, 125.0 / 50.0)); // 2.5
        assert!(approx(s.avg_trade_pct, (75.0 / 3.0) / 1000.0 * 100.0));
    }

    #[test]
    fn no_losing_trades_gives_infinite_profit_factor() {
        let s = compute(
            1000.0,
            &[1000.0],
            &[trade(10.0), trade(20.0)],
            252.0,
            0.0,
            1,
        );
        assert!(s.profit_factor.is_infinite());
    }

    #[test]
    fn degenerate_inputs_yield_zeros() {
        let s = compute(100.0, &[], &[], 252.0, 0.0, 0);
        assert_eq!(s.total_return_pct, 0.0);
        assert_eq!(s.sharpe, 0.0);
        assert_eq!(s.max_drawdown_pct, 0.0);
        assert_eq!(s.num_trades, 0);
        assert_eq!(s.profit_factor, 0.0);
        assert_eq!(s.exposure_pct, 0.0); // empty curve, no exposure
    }

    #[test]
    fn exposure_is_the_fraction_of_steps_in_a_position() {
        // 4 steps recorded, 3 of them holding a position -> 75% exposure
        let s = compute(100.0, &[100.0, 101.0, 102.0, 103.0], &[], 252.0, 0.0, 3);
        assert!(approx(s.exposure_pct, 75.0));
        assert!(approx(s.net_profit_pct, s.total_return_pct));
    }

    fn trade_with_fee(pnl: f64, fees: f64) -> Trade {
        Trade {
            fees,
            net_pnl: pnl - fees,
            ..trade(pnl)
        }
    }

    #[test]
    fn trade_metrics_are_net_of_the_recorded_fee() {
        // gross the run looks perfect; net it is a loser, and the metrics must say so
        let trades = [
            trade_with_fee(1.0, 3.0),
            trade_with_fee(1.0, 3.0),
            trade_with_fee(1.0, 3.0),
        ];
        let s = compute(1000.0, &[994.0], &trades, 252.0, 0.0, 1);
        assert_eq!(s.win_rate, 0.0); // every trade lost 2 after its fee
        assert_eq!(s.profit_factor, 0.0); // no net profit at all
        assert!(approx(s.avg_trade_pct, -2.0 / 1000.0 * 100.0));
        assert!(s.total_return_pct < 0.0);
    }

    #[test]
    fn a_non_positive_annualization_does_not_produce_nan() {
        // risk_free / periods_per_year would be 0/0; ADR 0007 says the set is always
        // defined, so a degenerate annualization falls back rather than propagating
        for ppy in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let s = compute(100.0, &[110.0, 99.0, 121.0], &[], ppy, 0.02, 3);
            assert!(s.sharpe.is_finite(), "ppy {ppy} gave sharpe {}", s.sharpe);
            assert!(s.sortino.is_finite());
            assert!(s.volatility_pct.is_finite());
            assert!(s.cagr_pct.is_finite());
            assert!(s.calmar.is_finite());
        }
    }

    #[test]
    fn drawdown_is_capped_at_a_hundred_percent_so_calmar_stays_ordered() {
        // bad debt takes equity below zero; an uncapped drawdown gave the deeper
        // bust the larger divisor and therefore the BETTER calmar (ADR 0029)
        let shallow = compute(100.0, &[100.0, 0.0], &[], 252.0, 0.0, 2);
        let deep = compute(100.0, &[100.0, -80.0], &[], 252.0, 0.0, 2);
        assert!(approx(shallow.max_drawdown_pct, 100.0));
        assert!(approx(deep.max_drawdown_pct, 100.0));
        assert!(
            deep.calmar <= shallow.calmar,
            "deeper bust scored {} against {}",
            deep.calmar,
            shallow.calmar
        );
    }

    #[test]
    fn a_bust_past_zero_yields_finite_stats_not_nan() {
        // equity goes negative (a perp bust past zero) and stays negative; CAGR
        // floors at -100%, post-bust returns are zeroed, nothing is NaN (ADR 0007)
        let s = compute(100.0, &[50.0, -20.0, -10.0], &[], 252.0, 0.0, 3);
        assert!(s.total_return_pct.is_finite());
        assert!(s.cagr_pct.is_finite());
        assert!(s.calmar.is_finite());
        assert!(s.sharpe.is_finite());
        assert!(s.sortino.is_finite());
        assert!(approx(s.cagr_pct, -100.0)); // a wipeout annualizes to -100%
    }
}
