//! In-core strategies for the compiled/batch path: a `Strategy` the engine drives
//! itself, GIL-free, calling `next` each bar. This is the loop-owning path (like a
//! classic backtester), distinct from the own-the-loop and RL paths where the
//! caller drives the engine.

use emsl_core::State;

use crate::engine::Engine;

/// A strategy the engine drives. `init` runs once at the start; `next` runs each
/// bar, seeing the current state and placing orders on the engine. Orders placed
/// in `next` fill on the following bar (no same-bar lookahead). `Send` so a batch
/// sweep can run strategies across threads.
pub trait Strategy: Send {
    /// Called once, before the first step.
    fn init(&mut self, _state: &State) {}

    /// Called each bar with the current state; place orders on `engine`.
    fn next(&mut self, state: &State, engine: &mut Engine);
}

/// Drive `engine` with `strategy` to the end of the data, returning the final
/// state. The strategy sees each bar and places orders that fill on the next.
pub fn run(engine: &mut Engine, strategy: &mut dyn Strategy) -> State {
    let mut state = engine.reset();
    strategy.init(&state);
    while !engine.done() {
        strategy.next(&state, engine);
        state = engine.step();
    }
    state
}

#[cfg(test)]
mod tests {
    use super::{run, Strategy};
    use crate::candles::Candles;
    use crate::engine::{Engine, EngineConfig};
    use emsl_core::{Candle, Market, State};

    fn series() -> Candles {
        Candles::new(vec![
            Candle {
                open: 100.0,
                high: 160.0,
                low: 90.0,
                close: 150.0,
                volume: 1000.0,
            },
            Candle {
                open: 200.0,
                high: 260.0,
                low: 190.0,
                close: 250.0,
                volume: 1000.0,
            },
            Candle {
                open: 300.0,
                high: 360.0,
                low: 290.0,
                close: 350.0,
                volume: 1000.0,
            },
        ])
    }

    fn cfg() -> EngineConfig {
        EngineConfig {
            market: Market::Spot,
            quote: 10_000.0,
            fee_taker: 0.0,
            fee_maker: 0.0,
            slippage_bps: 0.0,
            max_fill_fraction: 1.0,
            max_open_orders: 8,
            report: false,
            max_leverage: 0.0,
            impact: 0.0,
            funding_rate: 0.0,
            funding_interval: 0,
        }
    }

    struct BuyAndHold {
        bought: bool,
    }

    impl Strategy for BuyAndHold {
        fn next(&mut self, state: &State, engine: &mut Engine) {
            if !self.bought && state.position == 0.0 {
                engine.market_buy(1.0);
                self.bought = true;
            }
        }
    }

    #[test]
    fn run_drives_a_strategy_to_the_end() {
        let mut e = Engine::new(series(), cfg());
        let mut s = BuyAndHold { bought: false };
        let final_state = run(&mut e, &mut s);
        assert!(e.done());
        assert_eq!(final_state.position, 1.0); // bought once, then held
    }
}
