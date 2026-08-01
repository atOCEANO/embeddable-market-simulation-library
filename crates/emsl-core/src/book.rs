//! A fixed-slot resting order book. Only limit and stop orders rest here; market
//! orders fill immediately and never occupy a slot. Beyond capacity a new order
//! is rejected and handed back. Orders keep the id the engine assigned them, so
//! id assignment lives in one place, the engine.

use crate::order::{Order, OrderId};

/// A bounded set of resting orders held in a fixed number of slots, so the book
/// itself never grows or reallocates once built. Resolution follows slot order,
/// and `place` reuses the first free slot, so a cancelled slot hands its priority
/// to the next order placed (ADR 0006). Slots are allocated up front, so the
/// caller's capacity is bounded at the Python boundary (ADR 0027).
#[derive(Clone, Debug)]
pub struct RestingOrderBook {
    slots: Vec<Option<Order>>,
}

impl RestingOrderBook {
    /// A book with `capacity` slots.
    pub fn new(capacity: usize) -> RestingOrderBook {
        RestingOrderBook {
            slots: vec![None; capacity],
        }
    }

    /// Number of slots.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Number of resting orders.
    pub fn len(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// True when no order rests.
    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(Option::is_none)
    }

    /// True when every slot is taken.
    pub fn is_full(&self) -> bool {
        self.slots.iter().all(Option::is_some)
    }

    /// Rest `order` in a free slot, keeping its id. Returns the id, or hands the
    /// order back in `Err` when the book is full.
    pub fn place(&mut self, order: Order) -> Result<OrderId, Order> {
        match self.slots.iter_mut().find(|s| s.is_none()) {
            Some(slot) => {
                let id = order.id;
                *slot = Some(order);
                Ok(id)
            }
            None => Err(order),
        }
    }

    /// A resting order by id.
    pub fn get(&self, id: OrderId) -> Option<&Order> {
        self.slots.iter().flatten().find(|o| o.id == id)
    }

    /// A mutable resting order by id, for recording fills.
    pub fn get_mut(&mut self, id: OrderId) -> Option<&mut Order> {
        self.slots.iter_mut().flatten().find(|o| o.id == id)
    }

    /// Remove and return a resting order by id, freeing its slot.
    pub fn cancel(&mut self, id: OrderId) -> Option<Order> {
        for slot in &mut self.slots {
            if slot.as_ref().is_some_and(|o| o.id == id) {
                return slot.take();
            }
        }
        None
    }

    /// Remove every resting order.
    pub fn cancel_all(&mut self) {
        for slot in &mut self.slots {
            *slot = None;
        }
    }

    /// The resting orders, in slot order.
    pub fn iter(&self) -> impl Iterator<Item = &Order> {
        self.slots.iter().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::RestingOrderBook;
    use crate::enums::{Side, TimeInForce};
    use crate::order::{Order, OrderId};
    use crate::units::{Price, Qty};

    fn a_limit(id: u64) -> Order {
        Order::limit(
            OrderId(id),
            Side::Buy,
            Qty(1.0),
            Price(100.0),
            TimeInForce::Gtc,
            false,
            false,
        )
    }

    #[test]
    fn place_keeps_the_callers_id() {
        let mut b = RestingOrderBook::new(4);
        let id = b.place(a_limit(7)).unwrap();
        assert_eq!(id, OrderId(7));
        assert_eq!(b.get(OrderId(7)).unwrap().id, OrderId(7));
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn rejects_when_full() {
        let mut b = RestingOrderBook::new(2);
        b.place(a_limit(1)).unwrap();
        b.place(a_limit(2)).unwrap();
        assert!(b.is_full());
        assert!(b.place(a_limit(3)).is_err()); // rejected, order handed back
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn cancel_frees_a_slot_for_reuse() {
        let mut b = RestingOrderBook::new(1);
        let id = b.place(a_limit(1)).unwrap();
        assert!(b.is_full());
        assert!(b.cancel(id).is_some());
        assert!(b.is_empty());
        assert!(b.place(a_limit(2)).is_ok());
    }

    #[test]
    fn cancel_unknown_id_is_none() {
        let mut b = RestingOrderBook::new(2);
        b.place(a_limit(1)).unwrap();
        assert!(b.cancel(OrderId(999)).is_none());
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn cancel_all_empties_the_book() {
        let mut b = RestingOrderBook::new(4);
        b.place(a_limit(1)).unwrap();
        b.place(a_limit(2)).unwrap();
        b.cancel_all();
        assert!(b.is_empty());
        assert_eq!(b.iter().count(), 0);
    }

    #[test]
    fn get_mut_records_a_fill_in_place() {
        let mut b = RestingOrderBook::new(4);
        b.place(a_limit(5)).unwrap();
        b.get_mut(OrderId(5)).unwrap().filled = Qty(0.5);
        assert_eq!(b.get(OrderId(5)).unwrap().filled, Qty(0.5));
        assert!(b.get_mut(OrderId(999)).is_none());
    }

    #[test]
    fn place_reuses_the_first_freed_slot() {
        let mut b = RestingOrderBook::new(3);
        b.place(a_limit(1)).unwrap();
        b.place(a_limit(2)).unwrap();
        b.place(a_limit(3)).unwrap(); // slots [1, 2, 3]
        b.cancel(OrderId(2)).unwrap(); // free the middle slot
        b.place(a_limit(4)).unwrap(); // reuses the freed middle slot, not appended
        let ids: Vec<u64> = b.iter().map(|o| o.id.0).collect();
        assert_eq!(ids, vec![1, 4, 3]);
    }
}
