"""A deliberately slow, obviously correct bar simulator, written from the ADRs.

It exists to disagree with the engine. So it is written from the rules as the
decisions page states them and NOT from `engine.rs`: reading the implementation
while writing this would reproduce whatever the implementation does, including
its mistakes, and the whole value here is being an independent second opinion.

It is not fast and it is not clever. Every step is the sentence from the ADR it
implements, in the order the ADR puts it, with the number of the decision beside
it so a disagreement can be argued about rather than guessed at.

Scope: market, limit and stop orders on spot and perp, with fees, slippage,
market impact, the volume cap, the spot clamps, the perp margin cap, funding and
liquidation. ``start`` begins an episode part way into the series, which is what
the batched path does. Not modelled: FOK, post_only, replace, partial-fill
accumulation across bars for limits (the harness places one order at a time).
"""

import math

DUST = 1e-9          # ADR 0023: a fill at or below this moves neither cash nor position
MAX_SLIP = 0.99      # ADR 0024: the total adverse slip is held just under one


class Reference:
    def __init__(self, bars, market="spot", quote=10_000.0, fee_taker=0.0006,
                 fee_maker=0.0002, slippage_bps=0.0, max_fill_fraction=1.0,
                 leverage=10.0, impact=0.0, funding_rate=0.0, funding_interval=0,
                 max_open_orders=8, start=0):
        self.bars = bars
        self.market = market
        self.quote = float(quote)
        self.fee_taker = float(fee_taker)
        self.fee_maker = float(fee_maker)
        self.slip = float(slippage_bps) / 10_000.0
        self.cap = float(max_fill_fraction)
        self.leverage = float(leverage)
        self.impact = float(impact)
        self.funding_rate = float(funding_rate)
        self.funding_interval = int(funding_interval)
        # ADR 0028 bounds the resting book and ADR 0047 the pending queue, and once
        # either is full every further placement is REJECTED rather than queued.
        # Leaving it unbounded here was the second thing this reference got wrong:
        # a GTC limit rests forever (ADR 0016), so a run that keeps placing them
        # fills the book, and after that the two simulators are taking different
        # orders rather than disagreeing about the same ones
        self.slots = int(max_open_orders)

        self.position = 0.0
        self.entry = 0.0
        self.realized = 0.0
        self.open_fee = 0.0          # ADR 0030: the entry-side fee of what is held
        # ADRs 0002 and 0017: the funding schedule belongs to the SERIES, so an
        # episode started part way in funds the same absolute bars every other
        # episode does. Which is why an offset start is a starting tick over the
        # whole series here, and not a slice of it: `bars[start:]` would fund on
        # bars counted from the episode, and the two readings only part company
        # once something starts somewhere other than zero (ADR 0018)
        self.entry_tick = int(start)
        self.tick = int(start)
        self.fills = 0
        self.funding_paid = 0.0
        self.bust = False
        self.trades = []
        self.pending = []            # market orders, filling at the next open
        # ADR 0006: the book is SLOTS, not a queue. A freed slot is reused by the
        # next placement, so an order placed later can resolve before one placed
        # earlier, and that decides which of them gets the room when a cash or
        # margin clamp binds. Modelling it as a list was the third thing this
        # reference got wrong, and the one that took a differential run to see:
        # it only shows once something has filled and freed a slot
        self.resting = [None] * self.slots
        self.next_id = 1

    def free_slot(self):
        for i, order in enumerate(self.resting):
            if order is None:
                return i
        return None

    # ---------------------------------------------------------------- reading

    def equity(self, mark):
        if self.market == "spot":
            return self.quote + self.position * mark
        return self.quote + self.position * (mark - self.entry)

    # ---------------------------------------------------------------- placing

    def market_order(self, side, size, reduce_only=False, fok=False):
        # ADR 0070: a size that can never fill is refused a slot rather than taking one
        if not math.isfinite(size) or size <= 0.0:
            return None
        if len(self.pending) >= self.slots:
            return None
        order = dict(id=self.next_id, side=side, size=size, kind="market",
                     reduce_only=reduce_only, fok=fok)
        self.next_id += 1
        self.pending.append(order)
        return order["id"]

    def cancel(self, order_id):
        for i, order in enumerate(self.resting):
            if order is not None and order["id"] == order_id:
                self.resting[i] = None
                return True
        return False

    def cancel_all(self):
        """Every RESTING order. A pending market order is not resting and fills on
        the next step regardless, which is the distinction worth keeping."""
        dropped = sum(1 for order in self.resting if order is not None)
        self.resting = [None] * self.slots
        return dropped

    def replace(self, order_id, size=None, price=None, trigger=None):
        """ADR 0032: cancel and re-rest, and place NOTHING when the id is gone.

        ADR 0069 adds the half that took a defect to find: when the replacement is
        itself refused, the original goes back, because `None` means nothing
        happened and a caller trailing a stop reads it as "it already filled".
        """
        old = None
        slot = None
        for i, order in enumerate(self.resting):
            if order is not None and order["id"] == order_id:
                old, slot = order, i
                break
        if old is None:
            return None
        self.resting[slot] = None
        size = old["size"] - old["filled"] if size is None else size
        if old["kind"] == "limit":
            # the tif rides along with the side and the flags: a replacement is the
            # same order moved, so moving an IOC must not quietly make it a GTC
            placed = self.limit_order(old["side"], size,
                                      old["price"] if price is None else price,
                                      post_only=old["post_only"], tif=old["tif"],
                                      reduce_only=old["reduce_only"])
        else:
            placed = self.stop_order(old["side"], size,
                                     old["trigger"] if trigger is None else trigger,
                                     old["reduce_only"])
        if placed is None:
            self.resting[slot] = old
        return placed

    def limit_order(self, side, size, price, post_only=False, tif="gtc",
                    reduce_only=False):
        if not math.isfinite(size) or size <= 0.0 or not math.isfinite(price):
            return None
        # ADR 0045 by way of order.rs: a post_only limit that would cross the
        # market as it stands is REFUSED rather than turned taker
        if post_only:
            reference = self.bars[self.tick][3]
            crosses = price >= reference if side == "buy" else price <= reference
            if crosses:
                return None
        slot = self.free_slot()
        if slot is None:
            return None
        order = dict(id=self.next_id, side=side, size=size, kind="limit",
                     price=price, filled=0.0, reduce_only=reduce_only,
                     post_only=post_only, fok=(tif == "fok"), tif=tif)
        self.next_id += 1
        self.resting[slot] = order
        return order["id"]

    def stop_order(self, side, size, trigger, reduce_only=True):
        if not math.isfinite(size) or size <= 0.0 or not math.isfinite(trigger):
            return None
        slot = self.free_slot()
        if slot is None:
            return None
        order = dict(id=self.next_id, side=side, size=size, kind="stop",
                     trigger=trigger, filled=0.0, reduce_only=reduce_only,
                     post_only=False, fok=False, tif="gtc")
        self.next_id += 1
        self.resting[slot] = order
        return order["id"]

    # ----------------------------------------------------------------- prices

    def taker_slip(self, size, volume):
        # ADR 0004 flat slippage, ADR 0013 impact proportional to the bar taken,
        # ADR 0024 both adverse and the total held under one
        flat = max(self.slip, 0.0)
        extra = max(self.impact, 0.0) * (size / volume) if volume > 0.0 else 0.0
        total = flat + extra
        if not math.isfinite(total):
            return MAX_SLIP
        return min(total, MAX_SLIP)

    def within(self, side, price, bar):
        # ADR 0074: a taker price never leaves the bar that produced it
        return min(price, bar[1]) if side == "buy" else max(price, bar[2])

    def fillable(self, remaining, bar):
        # ADR 0005: one order takes at most this fraction of the bar's volume
        available = self.cap * bar[4]
        if not math.isfinite(remaining) or not math.isfinite(available) or available <= 0.0:
            return 0.0
        return min(remaining, available)

    # ---------------------------------------------------------------- clamping

    def clamp(self, side, size, price, reduce_only, is_taker=True):
        """Every risk rule, applied in the order the decisions state them."""
        # ADR 0070: neither a non-trade size nor a non-trade price is a fill
        if not math.isfinite(size) or size <= 0.0:
            return 0.0
        if not math.isfinite(price) or price <= 0.0:
            return 0.0

        signed = size if side == "buy" else -size

        # ADR 0012 and 0026: a perp's notional is capped at leverage times equity,
        # an over-cap position may be kept but not grown, a flip must fit outright,
        # and with no equity nothing may open
        if self.market == "perp" and self.leverage > 0.0:
            equity = self.equity(price)
            allowed = self.leverage * equity / price if equity > 0.0 else 0.0
            after = self.position + signed
            flipping = (self.position != 0.0 and after != 0.0
                        and (after > 0.0) != (self.position > 0.0))
            ceiling = allowed if flipping else max(allowed, abs(self.position))
            if abs(after) > ceiling:
                size = abs(ceiling * (1.0 if after > 0.0 else -1.0) - self.position)
                signed = size if side == "buy" else -size

        # ADR 0015: spot cannot short, so a sell is clamped to what is held
        if self.market == "spot" and side == "sell":
            size = min(size, max(self.position, 0.0))

        # ADR 0018: a spot buy fills only what the quote balance affords, fee
        # included, and the fee is the one this fill will actually be charged: a
        # resting limit is a maker, and using the taker rate here bought fractionally
        # less than the balance affords
        if self.market == "spot" and side == "buy":
            rate = self.fee_taker if is_taker else self.fee_maker
            per_unit = price * (1.0 + rate)
            affordable = self.quote / per_unit if per_unit > 0.0 else 0.0
            size = min(size, max(affordable, 0.0))

        # ADR 0028: reduce_only may only shrink, and never past flat
        if reduce_only:
            reduces = ((self.position > 0.0 and side == "sell")
                       or (self.position < 0.0 and side == "buy"))
            if not reduces:
                return 0.0
            size = min(size, abs(self.position))

        if not math.isfinite(size) or size <= DUST:
            return 0.0
        return size

    # ----------------------------------------------------------------- booking

    def apply(self, side, size, price, is_taker, liquidated=False):
        """ADR 0001 flip splitting, ADR 0030 the round-trip fee, ADR 0009 the log."""
        size = float(size)
        if size <= DUST:
            return 0.0
        rate = self.fee_taker if is_taker else self.fee_maker
        fee = size * price * rate
        signed = size if side == "buy" else -size
        before = self.position
        after = before + signed

        closed = 0.0
        if before != 0.0 and (before > 0.0) != (signed > 0.0):
            closed = min(size, abs(before))

        booked = 0.0
        if closed > DUST:
            direction = 1.0 if before > 0.0 else -1.0
            pnl = closed * (price - self.entry) * direction
            self.realized += pnl
            booked = pnl
            share = self.open_fee * (closed / abs(before))
            exit_fee = fee * (closed / size)
            self.trades.append(dict(
                entry_tick=self.entry_tick, exit_tick=self.tick,
                side="buy" if before > 0.0 else "sell", size=closed,
                entry_price=self.entry, exit_price=price,
                fees=share + exit_fee, pnl=pnl, net_pnl=pnl - share - exit_fee,
                bars_held=self.tick - self.entry_tick, liquidated=liquidated,
            ))
            self.open_fee -= share

        opened = size - closed
        if abs(after) <= DUST:
            after, self.entry, self.open_fee = 0.0, 0.0, 0.0
        elif closed > DUST and opened > DUST:
            # ADR 0001: the remainder opens fresh at the fill price
            self.entry = price
            self.open_fee = fee * (opened / size)
            self.entry_tick = self.tick
        elif opened > DUST and before == 0.0:
            self.entry = price
            self.open_fee = fee * (opened / size)
            self.entry_tick = self.tick
        elif opened > DUST:
            # a same-side add volume weights the entry (ADR 0001) and keeps its clock
            self.entry = (self.entry * abs(before) + price * opened) / abs(after)
            self.open_fee += fee * (opened / size)

        # ADR 0023: cash moves only with the position, and pays the fee with it.
        # The two markets book it differently and conflating them is the first
        # thing this reference got wrong: on SPOT the base is bought and sold, so
        # the notional leaves and returns; on a PERP nothing is bought, so opening
        # and marking never move quote and only realized PnL and fees do
        if self.market == "spot":
            self.quote -= signed * price + fee
        else:
            self.quote += booked - fee
        self.position = after
        self.fills += 1
        return size

    # -------------------------------------------------------------- the bar

    def bankruptcy_price(self):
        """ADR 0052: where closing the whole position leaves exactly nothing."""
        qty, size = self.position, abs(self.position)
        if size <= DUST:
            return None
        denominator = qty - size * self.fee_taker
        if abs(denominator) <= 2.2e-16:
            return None
        price = (qty * self.entry - self.quote) / denominator
        return price if math.isfinite(price) and price > 0.0 else None

    def step(self):
        if self.tick + 1 >= len(self.bars):
            return
        self.tick += 1
        bar = self.bars[self.tick]
        o, h, l, c, v = bar

        # ADR 0067: the liquidation is a price on this bar's path, so it is worked
        # out before any order resolves and the bar is clipped at it
        fence = None
        if self.market == "perp" and abs(self.position) > DUST:
            adverse = l if self.position > 0.0 else h
            if self.equity(adverse) <= 0.0:
                fence = self.bankruptcy_price()

        long = self.position >= 0.0
        gapped = fence is not None and (o <= fence if long else o >= fence)
        if gapped:
            self.liquidate(fence)

        if fence is not None and not gapped:
            clip = (lambda p: max(p, fence)) if long else (lambda p: min(p, fence))
            o, h, l, c = clip(o), clip(h), clip(l), clip(c)
        seen = (o, h, l, c, v)

        # 1. pending market orders fill at the open (ADR 0065)
        for order in self.pending:
            size = self.fillable(order["size"], seen)
            if size <= 0.0:
                continue
            slip = self.taker_slip(size, v)
            raw = o * (1.0 + slip) if order["side"] == "buy" else o * (1.0 - slip)
            price = self.within(order["side"], raw, seen)
            size = self.clamp(order["side"], size, price, order["reduce_only"], True)
            # ADR 0025: all or nothing is decided AFTER the clamps, because the
            # size that can actually fill is not the one the fill model offered:
            # the margin cap, the cash and short clamps and reduce_only all shrink
            # it afterwards, so a FOK that could only fill in part books nothing
            if order["fok"] and size + DUST < order["size"]:
                continue
            self.apply(order["side"], size, price, True)
        self.pending = []           # ADR 0016: a market order is IOC

        # 2. resting orders, worst exit first among the exits (ADRs 0056, 0071)
        offers = []
        for order in [x for x in self.resting if x is not None]:
            remaining = order["size"] - order["filled"]
            if remaining <= DUST:
                continue
            if order["kind"] == "limit":
                touched = (l <= order["price"]) if order["side"] == "buy" else (h >= order["price"])
                if not touched:
                    continue
                size = self.fillable(remaining, seen)
                if size <= 0.0:
                    continue
                # ADR 0045: it fills AT the limit and pays taker if it was already
                # through the market when the bar opened
                crossed = (order["price"] >= o) if order["side"] == "buy" else (order["price"] <= o)
                offers.append((order, size, order["price"], crossed))
            else:
                hit = (h >= order["trigger"]) if order["side"] == "buy" else (l <= order["trigger"])
                if not hit:
                    continue
                size = self.fillable(remaining, seen)
                if size <= 0.0:
                    continue
                base = max(o, order["trigger"]) if order["side"] == "buy" else min(o, order["trigger"])
                slip = self.taker_slip(size, v)
                raw = base * (1.0 + slip) if order["side"] == "buy" else base * (1.0 - slip)
                offers.append((order, size, self.within(order["side"], raw, seen), True))

        offers = self.worst_first(offers)
        for order, size, price, taker in offers:
            got = self.clamp(order["side"], size, price, order["reduce_only"], taker)
            # judged against the account as it stands when this order is reached,
            # not against a snapshot taken before the bar's earlier fills (ADR 0025)
            if order["fok"] and got + DUST < order["size"] - order["filled"]:
                continue
            applied = self.apply(order["side"], got, price, taker)
            # ADR 0068: consumed by what it filled, and it rests until there is none left
            order["filled"] += applied
        for i, order in enumerate(self.resting):
            if order is not None and order["size"] - order["filled"] <= DUST:
                self.resting[i] = None
        # ADR 0016: an IOC or FOK limit gets ONE bar of exposure and is cancelled
        # after it, filled or not, where a GTC limit waits for the next bar. This
        # is the whole difference between the three, and it lives here because a
        # partially filled IOC keeps its fill and loses only the remainder
        for i, order in enumerate(self.resting):
            if order is not None and order["kind"] == "limit" and order["tif"] != "gtc":
                self.resting[i] = None

        # 3. funding, then the liquidation check (ADRs 0002, 0017)
        if self.market == "perp" and self.funding_interval > 0 and self.tick % self.funding_interval == 0:
            payment = self.position * c * self.funding_rate
            self.quote -= payment
            self.funding_paid += payment

        # 4. liquidation at the bar's adverse extreme (ADR 0003)
        if self.market == "perp" and abs(self.position) > DUST:
            adverse = bar[2] if self.position > 0.0 else bar[1]
            if self.equity(adverse) <= 0.0:
                price = self.bankruptcy_price()
                if price is not None:
                    self.liquidate(price)
                self.bust = True

        if self.equity(bar[3]) <= 0.0:      # ADR 0019
            self.bust = True

    def worst_first(self, offers):
        """ADR 0056 as corrected by 0071: only the exits reorder, among their own slots."""
        position, entry = self.position, self.entry

        def adversity(item):
            order, _, price, _ = item
            reduces = ((position > 0.0 and order["side"] == "sell")
                       or (position < 0.0 and order["side"] == "buy"))
            if not reduces:
                return math.inf
            return (price - entry) * (1.0 if position > 0.0 else -1.0)

        slots = [i for i, x in enumerate(offers) if math.isfinite(adversity(x))]
        if len(slots) < 2:
            return offers
        ordered = sorted(slots, key=lambda i: adversity(offers[i]))
        out = list(offers)
        for slot, take in zip(slots, ordered):
            out[slot] = offers[take]
        return out

    def liquidate(self, price):
        size = abs(self.position)
        if size <= DUST:
            return
        side = "sell" if self.position > 0.0 else "buy"
        self.apply(side, size, price, True, liquidated=True)
        self.bust = True
