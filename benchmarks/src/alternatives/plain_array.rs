use std::cmp::Ordering;

pub struct PlainArray<T, C>
where
    C: Fn(&T, &T) -> Ordering,
{
    comparator: C,
    data: Vec<T>,
}

#[allow(dead_code)]
impl<T, C> PlainArray<T, C>
where
    C: Fn(&T, &T) -> Ordering,
    T: Clone + std::fmt::Debug,
{
    pub fn new(comparator: C, capacity: usize) -> PlainArray<T, C> {
        PlainArray {
            comparator,
            data: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn insert(&mut self, t: T) -> bool {
        match self.data.binary_search_by(|x| (self.comparator)(x, &t)) {
            Ok(_) => {
                false
            }
            Err(idx) => {
                if idx < self.data.len() {
                    self.data.insert(idx, t);
                } else {
                    self.data.push(t);
                }
                true
            }
        }
    }

    pub fn remove(&mut self, t: &T) -> bool {
        match self.data.binary_search_by(|x| (self.comparator)(x, &t)) {
            Ok(idx) => {
                self.data.remove(idx);
                true
            }
            Err(_) => {
                false
            }
        }
    }

    pub fn collect(&self) -> Vec<T> {
        self.data.clone()
    }

}
