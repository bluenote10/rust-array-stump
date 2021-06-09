use std::cmp::Ordering;

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub struct Index {
    outer: usize,
    inner: usize,
}

impl Index {
    pub const FIRST: Index = Index{outer: 0, inner: 0};

    pub fn new(outer: usize, inner: usize) -> Index {
        Index{outer, inner}
    }
}

#[derive(Clone, Copy, Debug)]
pub struct IndexTransition {
    old: Index,
    new: Index,
}

impl IndexTransition {
    pub fn new(old: Index, new: Index) -> IndexTransition {
        IndexTransition{old, new}
    }
}

/// The core data structure representing a two-level sorted stump.
pub struct ArrayStump<T, C>
where
    C: Fn(&T, &T) -> Ordering,
{
    comparator: C,
    data: Vec<Vec<T>>,
    init_capacity: u16,
    capacity: u16,
    num_elements: usize,
}

impl<T, C> ArrayStump<T, C>
where
    C: Fn(&T, &T) -> Ordering,
    T: Clone,
    T: std::fmt::Debug, // TODO: remove soon
{
    /// Creates a new `ArrayStump` instance.
    pub fn new(comparator: C) -> ArrayStump<T, C> {
        ArrayStump::new_explicit(comparator, 512)
    }

    /// Creates a new `ArrayStump` instance with explicit control over internal parameters.
    pub fn new_explicit(comparator: C, init_capacity: u16) -> ArrayStump<T, C> {
        let data = Vec::with_capacity(init_capacity as usize);
        ArrayStump {
            comparator,
            data,
            init_capacity,
            capacity: 8,
            num_elements: 0,
        }
    }

    /// Returns the length (i.e., number of elements stored).
    pub fn len(&self) -> usize {
        self.num_elements
    }

    pub fn is_empty(&self) -> bool {
        self.num_elements == 0
    }

    /// Insert a value.
    pub fn insert(&mut self, t: T) -> Option<Index> {
        if self.data.is_empty() {
            self.data.push(self.new_block(t));
            self.num_elements += 1;
            return Some(Index::new(0, 0));
        }

        // Binary search for block index
        let (idx_block, equals) = binary_search_by(
            &self.data,
            |block| (self.comparator)(&block[0], &t),
        );
        if equals {
            return None;
        }

        // Convert from "first larger" to "last smaller" index semantics
        let mut idx_block = if idx_block > 0 {
            idx_block - 1
        } else {
            0
        };

        // Split block if necessary
        if self.data[idx_block].len() >= self.capacity as usize {
            let tail_from = (self.capacity / 2) as usize;
            let tail_upto = self.capacity as usize;
            let block_tail = self.data[idx_block][tail_from .. tail_upto].to_vec();

            // Note: The `.to_vec()` requires T: Clone but is faster than using drain. Keep?
            // let block_tail: Vec<_> = self.data[idx_block].drain(tail_from .. tail_upto).collect();

            // Note: why not use Vec.split_off?

            self.data[idx_block].truncate(tail_from);
            self.data.insert(idx_block + 1, block_tail);

            // Determine into which of the two split blocks the new value goes.
            let cmp = (self.comparator)(&t, &self.data[idx_block + 1][0]);
            if cmp == Ordering::Equal {
                return None;
            } else if cmp == Ordering::Greater {
                idx_block += 1;
            }
        }

        // Binary search for value index
        let (idx_value, equals) = binary_search_by(
            &self.data[idx_block],
            |x| (self.comparator)(&x, &t),
        );
        if equals {
            return None;
        }

        // Value insert
        let block_len = self.data[idx_block].len();
        if idx_value < block_len {
            self.data[idx_block].insert(idx_value, t);
        } else {
            self.data[idx_block].push(t);
        }

        self.num_elements += 1;

        if self.data.len() > self.capacity as usize * 5 {
            self.capacity *= 2;
        }

        Some(Index::new(idx_block, idx_value))
    }

    /// Remove a value.
    pub fn remove(&mut self, t: &T) -> bool {
        if self.data.is_empty() {
            return false;
        }

        if let Some(idx) = self.find(t) {
            self.remove_by_index(idx);
            true
        } else {
            false
        }
    }

    /// Remove a value by its index.
    #[inline]
    pub fn remove_by_index(&mut self, idx: Index) {
        let idx_block = idx.outer;
        let idx_value = idx.inner;

        if self.data[idx_block].len() > 1 {
            self.data[idx_block].remove(idx_value);
        } else {
            self.data.remove(idx_block);
        }
        self.num_elements -= 1;

        if self.get_leaf_fill_ratio() < 0.1 && self.capacity > 2 {
            self.capacity /= 2;
            apply_reduced_capacity(&mut self.data, self.capacity);
        }
    }

    /// Try to find an existing value.
    #[inline]
    pub fn find(&self, t: &T) -> Option<Index> {
        if self.data.is_empty() {
            return None;
        }

        // Binary search for block index
        let (idx_block, equals) = binary_search_by(
            &self.data,
            #[inline] |block| (self.comparator)(&block[0], &t),
        );
        if equals {
            return Some(Index{outer: idx_block, inner: 0});
        }

        // Convert from "first larger" to "last smaller" index semantics
        let idx_block = if idx_block > 0 {
            // TODO: Probably we can short circuit here, because if the element is smaller
            // then the first element of the first block, we don't have to do binary search.
            // Needs tests...
            idx_block - 1
        } else {
            0
        };

        // Binary search for value index
        let (idx_value, equals) = binary_search_by(
            &self.data[idx_block],
            #[inline] |x| (self.comparator)(&x, &t),
        );
        if equals {
            return Some(Index{outer: idx_block, inner: idx_value});
        }

        None
    }

    /// Returns the element at a given index.
    /// Caution: An index that has been obtained before mutating the data structure is an
    /// invalid index. Calling this function with an invalid index may panic with index
    /// out of bounds.
    pub fn get_by_index(&self, idx: Index) -> &T {
        &self.data[idx.outer][idx.inner]
    }

    /// Returns the index of the next element if there is one.
    pub fn next_index(&self, idx: Index) -> Option<Index> {
        if idx.outer >= self.data.len() {
            None
        } else if idx.inner < self.data[idx.outer].len() - 1 {
            Some(Index::new(idx.outer, idx.inner + 1))
        } else if idx.outer < self.data.len() - 1 {
            Some(Index::new(idx.outer + 1, 0))
        } else {
            None
        }
    }

    /// Returns the index of the previous element if there is one.
    pub fn prev_index(&self, idx: Index) -> Option<Index> {
        if idx.outer >= self.data.len() {
            None
        } else if idx.inner > 0 {
            Some(Index::new(idx.outer, idx.inner - 1))
        } else if idx.outer > 0 {
            Some(Index::new(idx.outer - 1, self.data[idx.outer - 1].len() - 1))
        } else {
            None
        }
    }

    /// Returns the minimum value.
    pub fn min(&self) -> Option<&T> {
        if self.num_elements > 0 {
            Some(&self.data[0][0])
        } else {
            None
        }
    }

    /// Returns the maximum value.
    pub fn max(&self) -> Option<&T> {
        if self.num_elements > 0 {
            let i = self.data.len() - 1;
            let j = self.data[i].len() - 1;
            Some(&self.data[i][j])
        } else {
            None
        }
    }

    /// Traverse collection given a callback.
    pub fn traverse<F>(&self, mut f: F)
    where
        F: FnMut(usize, &T),
    {
        let mut i = 0;
        for block in &self.data {
            for x in block {
                f(i, x);
                i += 1;
            }
        }
    }

    /// Collect collection into a vector.
    pub fn collect(&self) -> Vec<T> {
        let mut data = Vec::with_capacity(self.num_elements);
        self.traverse(|_, x| data.push(x.clone()));
        data
    }

    fn new_block(&self, t: T) -> Vec<T> {
        let capacity = self.capacity.max(self.init_capacity);
        let mut block = Vec::with_capacity(capacity as usize);
        block.push(t);
        block
    }

    /// Get the average fill ratio of leafs, i.e., a value of 0.5 means that
    /// leafs are on average half full.
    ///
    /// This is an O(1) operation.
    pub fn get_leaf_fill_ratio(&self) -> f64 {
        (self.num_elements as f64) / (self.capacity as f64 * self.data.len() as f64)
    }

    /// Get the minimum number of elements in a leaf.
    ///
    /// This requires iterating all blocks, and thus, is an O(sqrt N) operation.
    pub fn get_leaf_fill_min(&self) -> Option<usize> {
        self.data.iter().map(|block| block.len()).min()
    }

    /// Get the maximum number of elements in a leaf.
    ///
    /// This requires iterating all blocks, and thus, is an O(sqrt N) operation.
    pub fn get_leaf_fill_max(&self) -> Option<usize> {
        self.data.iter().map(|block| block.len()).max()
    }

    /// Get the current max leaf capacity.
    pub fn get_capacity(&self) -> u16 {
        self.capacity
    }

    /// Get the current number of blocks.
    pub fn get_num_blocks(&self) -> usize {
        self.data.len()
    }

    /// Internal debug helper function.
    pub fn debug(&self) {
        println!("{:?}", self.data);
    }

    pub fn debug_order(&self) {
        println!("--- DEBUG ORDER");
        let mut remember : Option<&T> = None;
        for (idx, block) in self.data.iter().enumerate() {
            println!("-- BLOCK #{} ({}/{} elements)", idx, block.len(), block.capacity());
            for value in block {
                if let Some(last) = remember {
                    println!("{:?}", (self.comparator)(last, value));
                }
                println!("{:?}", value);
                remember = Some(value);
            }
        }
    }

    // possibility to fix an index after fixing the rank
    pub fn fix_index(&self, transitions: &[IndexTransition], idx: Index) -> Index {
        let mut result = idx;
        for transition in transitions {
            result = self.fix_index_single(transition, result);
        }
        result
    }

    fn fix_index_single(&self, transition: &IndexTransition, idx: Index) -> Index {
        let old = transition.old;
        let new = transition.new;

        if idx == old {
            new
        } else if old < new {
            if idx < old || new < idx {
                idx
            } else {
                self.prev_index(idx).unwrap()
            }
        } else if idx < new || old < idx  {
            idx
        } else {
            self.next_index(idx).unwrap()
        }
    }

    // sort element referenced by 'idx' back (at least after element referenced by 'after')
    fn sort_element_back(&mut self, idx: Index, after: Index) -> IndexTransition {
        let cmp = &self.comparator;
        let element = self.get_by_index(idx);
        let mut next = self.next_index(after);

        while let Some(nidx) = next {
            if cmp(element, self.get_by_index(nidx)) != Ordering::Greater {
                break;
            }
            next = self.next_index(nidx);
        }

        let dest = if let Some(next) = next {
            self.prev_index(next).unwrap()
        } else {
            let last_block = self.data.len() - 1;
            Index::new(last_block, self.data[last_block].len()-1)
        };

        let element = self.data[idx.outer].remove(idx.inner);

        for block_index in idx.outer..dest.outer {
            let crosser = self.data[block_index + 1].remove(0);
            self.data[block_index].push(crosser);
        }

        self.data[dest.outer].insert(dest.inner, element);

        IndexTransition::new( idx, dest )
    }

      // sort element referenced by 'idx' forward (at least before element referenced by 'before')
      fn sort_element_forward(&mut self, idx: Index, before: Index) -> IndexTransition {
        let cmp = &self.comparator;
        let element = self.get_by_index(idx);
        let mut next = self.prev_index(before);

        while let Some(nidx) = next {
            if cmp(self.get_by_index(nidx), element) != Ordering::Greater {
                break;
            }
            next = self.prev_index(nidx);
        }

        let dest = if let Some(next) = next {
            self.next_index(next).unwrap()
        } else {
            Index::FIRST
        };

        let element = self.data[idx.outer].remove(idx.inner);

        for block_index in (dest.outer..idx.outer).rev() {
            let block = &mut self.data[block_index];
            let crosser = block.remove(block.len() - 1);
            self.data[block_index + 1].insert(0, crosser);
        }
        self.data[dest.outer].insert(dest.inner, element);

        IndexTransition::new( idx, dest )
    }

    // ensure order for the elements in range [from, to] is correct
    pub fn fix_rank_range(&mut self, from: Index, to: Index) -> Vec<IndexTransition> {
        let mut result = vec![];

        // also check whether first element needs to be moved forward -> start one element early
        let mut current = self.prev_index(from).unwrap_or(from);

        // 'next_next' is the first element with known good order after the range to sort
        let next_next = self.next_index(to);

        while current <= to {
            let element = self.get_by_index(current);
            let next = self.next_index(current);

            if let Some(next) = next {
                // if there's an order violation
                if (self.comparator)(element, self.get_by_index(next)) == Ordering::Greater {
                    if let Some(next_next) = next_next {
                        // check whether 'current' needs to go back or 'next' needs to go forward
                        if (self.comparator)(element, self.get_by_index(next_next)) == Ordering::Greater {
                            result.push(self.sort_element_back(current, next_next));
                        } else {
                            result.push(self.sort_element_forward(next, current));
                        }
                    } else {
                        // special case when there is nothing after the range to sort
                        result.push(self.sort_element_back(current, to));
                    }
                }
                // advance
                current = next;
            } else {
                // we're done
                break;
            }
        }

        result
    }
}

// I'm not quite sure if we should support the Index trait, because index semantics
// are not the most natural operation on tree-like data structures. Maybe it is better
// to stick with `get_by_index` and `get_by_rank` to be explicit about the two possible
// semantics that indexing could have.
#[cfg(feature="indextrait")]
impl<T, C> std::ops::Index<Index> for ArrayStump<T, C>
where
    C: Fn(&T, &T) -> Ordering,
    T: Clone,
{
    type Output = T;
    /// Access an element via its index. This operation is only valid, if the data has not
    /// been modified since the index has been obtained.
    fn index<'a>(&'a self, i: Index) -> &'a T {
        &self.data[i.outer][i.inner]
    }
}

// Note: We are using our own implementation of binary search, because the implementation
// in the standard library is optimized for fast comparison functions, and requires more
// comparison function evaluations.
fn binary_search_by<T, F>(data: &[T], mut f: F) -> (usize, bool)
where
    F: FnMut(&T) -> Ordering,
    T: std::fmt::Debug,
{
    if data.is_empty() {
        return (data.len(), false);
    }
    let mut l: usize = 0;
    let mut r: usize = data.len();

    while r > l {
        let mid = l + (r - l) / 2;

        let mid_el = unsafe { &data.get_unchecked(mid) };
        // println!("{} {} {} {:?}", l, r, mid, mid_el);

        let cmp = f(mid_el);
        match cmp {
            Ordering::Greater => {
                r = mid;
            }
            Ordering::Equal => {
                return (mid, true)
            }
            Ordering::Less => {
                l = mid + 1;
            }
        }
    }

    (r, false)
}

#[inline]
fn get_elements_per_block(i: usize, len: usize, num_blocks: usize) -> usize {
    len / num_blocks + if i < (len % num_blocks) { 1 } else { 0 }
}

#[allow(dead_code)]
fn apply_reduced_capacity<T>(data: &mut Vec<Vec<T>>, new_capacity: u16)
where
    T: Clone,
{
    let new_capacity = new_capacity as usize;
    let mut i = 0;
    while i < data.len() {
        let len = data[i].len();
        if len <= new_capacity {
            i += 1;
        } else {
            let num_required_blocks = (len / new_capacity) + if len % new_capacity > 0 { 1 } else { 0 };
            if num_required_blocks == 2 {
                let divide = get_elements_per_block(0, len, num_required_blocks);

                let block_tail = data[i][divide .. ].to_vec();
                data[i].truncate(divide);
                data.insert(i + 1, block_tail);
                i += 2;
            } else {
                let original = data[i].clone();

                let mut divide = get_elements_per_block(0, len, num_required_blocks);
                data[i].truncate(divide);
                i += 1;

                for j in 1 .. num_required_blocks {
                    let next_divide = divide + get_elements_per_block(j, len, num_required_blocks);
                    let block = original[divide .. next_divide].to_vec();
                    data.insert(i, block);
                    i += 1;
                    divide = next_divide;
                }
            }
        }
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use std::cmp::Ordering;
    use pretty_assertions::assert_eq;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    macro_rules! vec2d {
        ($($x:expr),*) => {{
            let data = [ $(to_vec_i32(&$x)),* ].to_vec();
            data
        }}
    }

    fn int_comparator(a: &i32, b: &i32) -> Ordering {
        a.cmp(b)
    }

    // ------------------------------------------------------------------------
    // Binary search testing
    // ------------------------------------------------------------------------

    pub fn binary_search_by_reference<T, F>(data: &[T], mut f: F) -> (usize, bool)
    where
        F: FnMut(&T) -> Ordering,
    {
        #[allow(clippy::needless_range_loop)]
        for i in 0 .. data.len() {
            let x = &data[i];
            let cmp = f(x);
            match cmp {
                Ordering::Equal => return (i, true),
                Ordering::Greater => return (i, false),
                _ => {}
            }
        }
        (data.len(), false)
    }

    pub fn generate_random_array(rng: &mut StdRng, len: usize) -> (Vec<i32>, Vec<i32>) {
        let mut data = Vec::new();

        let mut last = 0;
        for _ in 0 .. len {
            data.push(last);
            if rng.gen::<bool>() {
                last += 1;
            }
        }

        let mut test_values = Vec::with_capacity(data.len() * 3);
        for x in &data {
            test_values.push(*x - 1);
            test_values.push(*x);
            test_values.push(*x + 1);
        }
        (data, test_values)
    }

    fn test_against_reference(data: &[i32], value: i32) {
        println!("{:?} {}", data, value);

        let (idx_actual, equals_actual) = binary_search_by(&data, |x| x.cmp(&value));
        let (idx_expect, equals_expect) = binary_search_by_reference(&data, |x| x.cmp(&value));

        assert_eq!(equals_actual, equals_expect);
        if !equals_expect {
            assert_eq!(idx_actual, idx_expect);
        } else {
            assert_eq!(data[idx_actual], value);
        }
    }

    #[test]
    fn test_binary_search_empty() {
        let data: Vec::<i32> = vec![];
        assert_eq!(binary_search_by(&data, |x| int_comparator(x, &0)), (0, false));
    }

    #[test]
    fn test_binary_search_basic() {
        let data = [1, 2, 3];
        assert_eq!(binary_search_by(&data, |x| int_comparator(x, &0)), (0, false));
        assert_eq!(binary_search_by(&data, |x| int_comparator(x, &1)), (0, true));
        assert_eq!(binary_search_by(&data, |x| int_comparator(x, &2)), (1, true));
        assert_eq!(binary_search_by(&data, |x| int_comparator(x, &3)), (2, true));
        assert_eq!(binary_search_by(&data, |x| int_comparator(x, &4)), (3, false));
    }

    #[test]
    fn test_binary_search_brute_force() {
        let num_random_variations = 100;
        let mut rng: StdRng = SeedableRng::seed_from_u64(0);
        for array_len in 0 ..= 32 {
            for _ in 0 .. num_random_variations {
                let (data, test_values) = generate_random_array(&mut rng, array_len);
                assert_eq!(data.len(), array_len);
                for value in &test_values {
                    test_against_reference(&data, *value);
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // Array tests
    // ------------------------------------------------------------------------

    macro_rules! new_array {
        ($capacity:expr, $data:expr) => {{
            let data: Vec<Vec<i32>> = $data;
            let num_elements = data.iter().map(|block| block.len()).sum();
            ArrayStump {
                comparator: int_comparator,
                init_capacity: 64,
                capacity: $capacity,
                data: $data,
                num_elements,
            }
        }};
    }
    macro_rules! insert_many {
        ($a:expr, $data:expr) => {
            for x in $data.iter() {
                $a.insert(x.clone());
            }
        };
    }

    #[test]
    fn test_array_stump_initial_push() {
        let mut a = new_array!(16, vec![]);
        assert_eq!(a.len(), 0);
        a.insert(0);
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn test_array_stump_prefers_push() {
        let mut a = new_array!(16, vec![vec![1, 2], vec![4, 5]]);
        assert_eq!(a.len(), 4);
        a.insert(3);
        assert_eq!(a.data, [vec![1, 2, 3], vec![4, 5]]);
        assert_eq!(a.len(), 5);
    }

    #[test]
    fn test_array_stump_no_index_hiccup() {
        let mut a = new_array!(8, vec![vec![2], vec![4], vec![6, 8]]);
        a.insert(7);
        assert_eq!(a.data, [vec![2], vec![4], vec![6, 7, 8]]);
    }

    #[test]
    fn test_split() {
        let mut a = new_array!(2, vec![vec![2, 4], vec![6, 8]]);
        assert_eq!(a.len(), 4);
        a.insert(1);
        assert_eq!(a.data, [vec![1, 2], vec![4], vec![6, 8]]);
        assert_eq!(a.len(), 5);

        let mut a = new_array!(2, vec![vec![2, 4], vec![6, 8]]);
        assert_eq!(a.len(), 4);
        a.insert(3);
        assert_eq!(a.data, [vec![2, 3], vec![4], vec![6, 8]]);
        assert_eq!(a.len(), 5);

        let mut a = new_array!(2, vec![vec![2, 4], vec![6, 8]]);
        assert_eq!(a.len(), 4);
        a.insert(5);
        assert_eq!(a.data, [vec![2], vec![4, 5], vec![6, 8]]);
        assert_eq!(a.len(), 5);

        let mut a = new_array!(2, vec![vec![2, 4], vec![6, 8]]);
        assert_eq!(a.len(), 4);
        a.insert(7);
        assert_eq!(a.data, [vec![2, 4], vec![6, 7], vec![8]]);
        assert_eq!(a.len(), 5);

        let mut a = new_array!(2, vec![vec![2, 4], vec![6, 8]]);
        assert_eq!(a.len(), 4);
        a.insert(9);
        assert_eq!(a.data, [vec![2, 4], vec![6], vec![8, 9]]);
        assert_eq!(a.len(), 5);
    }

    #[test]
    fn test_split_on_index_with_equality() {
        // We must make sure that the element at the split index has proper equality check
        let mut a = new_array!(8, vec![vec![5, 7, 11, 17, 19, 22, 29, 30]]);
        let equals = a.insert(19);
        assert_eq!(equals, None);  // no insertion
    }

    #[test]
    fn test_array_stump_collect() {
        for &cap in &[2, 3, 4, 5] {
            let mut a = ArrayStump::new_explicit(int_comparator, cap as u16);
            insert_many!(a, [1, 2, 3, 4]);
            assert_eq!(a.collect(), [1, 2, 3, 4]);
            assert_eq!(a.collect().len(), a.len());

            let mut a = ArrayStump::new_explicit(int_comparator, cap as u16);
            insert_many!(a, [1, 2, 3, 4]);
            assert_eq!(a.collect(), [1, 2, 3, 4]);
            assert_eq!(a.collect().len(), a.len());
        }
    }

    #[test]
    fn test_find() {
        let a = new_array!(16, vec![vec![2, 4], vec![6], vec![8]]);
        assert_eq!(a.find(&2), Some(Index::new(0, 0)));
        assert_eq!(a.find(&4), Some(Index::new(0, 1)));
        assert_eq!(a.find(&6), Some(Index::new(1, 0)));
        assert_eq!(a.find(&8), Some(Index::new(2, 0)));
        for x in [1, 3, 5, 7, 9].iter() {
            assert_eq!(a.find(x), None);
        }
    }

    #[test]
    fn test_remove() {
        let mut a = new_array!(16, vec![vec![2, 4], vec![6], vec![8]]);
        a.remove(&2);
        assert_eq!(a.collect(), vec![4, 6, 8]);
        let mut a = new_array!(16, vec![vec![2, 4], vec![6], vec![8]]);
        a.remove(&4);
        assert_eq!(a.collect(), vec![2, 6, 8]);
        let mut a = new_array!(16, vec![vec![2, 4], vec![6], vec![8]]);
        a.remove(&6);
        assert_eq!(a.collect(), vec![2, 4, 8]);
        let mut a = new_array!(16, vec![vec![2, 4], vec![6], vec![8]]);
        a.remove(&8);
        assert_eq!(a.collect(), vec![2, 4, 6]);
    }

    #[test]
    fn test_failing() {
        let mut at = ArrayStump::new_explicit(|a: &f64, b: &f64| a.partial_cmp(b).unwrap(), 16);
        let vals = vec![0.6994135560499647, 0.15138991083383901, 0.17989509662598502, 0.22855960374503625, 0.7394173591733456, 0.8606810583068278, 0.025843624735059523, 0.1416162372765526, 0.9789425643425963, 0.6312677864630949, 0.34678659888024466, 0.7876614416763924, 0.6260871506068197, 0.34733559592131624, 0.5722923635764159, 0.14416998787798063, 0.839158671060864, 0.2621428817535354, 0.9334439919690996, 0.016414089291711065, 0.8795903741012259, 0.051958655798298614, 0.8313985552845266, 0.026928982020677505, 0.779969564116276, 0.6437306675337413, 0.03822809941255523, 0.777911020749552, 0.4639770428538855, 0.7039388191038694, 0.31363729764551374, 0.8111651227165783, 0.5174339383176408, 0.49384841003283086, 0.5214549475595969, 0.0823716635367353, 0.7310183483079477, 0.6196297749276181, 0.6226877845880779, 0.8987550167723078, 0.9536731852226494, 0.2719858776118911, 0.837006810218081, 0.7570466272336563, 0.9649096907962248, 0.09547804495341239, 0.26299769639555115, 0.6883529379785718, 0.23545125345269502, 0.5611223421257663, 0.81145380876482, 0.7821846165410649, 0.8385374221326543, 0.2287909449815878, 0.9938012642875733, 0.30515950398348823, 0.021945251189301795, 0.7456118789178752, 0.24917873250483202, 0.19461925257672297, 0.08596890658908873, 0.8208413553993631, 0.2799020116906893, 0.622583855342935, 0.3406868767224045, 0.7125811318179431, 0.8171813899535424, 0.9875530622413784, 0.8124194427320398, 0.27890169087536465, 0.4582999489551358, 0.8170130026270258, 0.1116683852975886, 0.9523649049789342, 0.1626401579175366, 0.7006463636943299, 0.5396656897339597, 0.73824000529768, 0.8975902131523751, 0.3138666758196337, 0.959190654990596, 0.6786382471256971, 0.8807317907186307, 0.9923109213923168, 0.7704353170122445, 0.20331717853087872, 0.9191784945915048, 0.3458975102965529, 0.44567705127366397, 0.08758863415076357, 0.8940937525362007, 0.2046747373689708, 0.1540080303289173, 0.8088614347095653, 0.09821866105193844, 0.050284880746519045, 0.9585396829998039, 0.35100273069739263, 0.8263845327940142, 0.6305932414080216];
        for (i, x) in vals.iter().enumerate() {
            at.insert(*x);
            let mut expected = vals[0 .. i + 1].to_vec();
            expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(at.collect(), expected);
        }
    }

    // ------------------------------------------------------------------------
    // Array tests -- misc functionality
    // ------------------------------------------------------------------------

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_statistics_and_debugging() {
        let a = new_array!(4, vec2d![[1], [2, 3], [4, 5, 6]]);
        a.debug();
        assert_eq!(a.get_leaf_fill_min().unwrap(), 1);
        assert_eq!(a.get_leaf_fill_max().unwrap(), 3);
        assert_eq!(a.get_leaf_fill_ratio(), 0.5);
        assert_eq!(a.get_num_blocks(), 3);
        assert_eq!(a.get_capacity(), 4);
    }

    // ------------------------------------------------------------------------
    // Capacity adaptation
    // ------------------------------------------------------------------------

    fn to_vec_i32(a: &[i32]) -> Vec<i32> {
        a.to_vec()
    }

    #[test]
    fn test_get_elements_per_block() {
        assert_eq!(get_elements_per_block(0, 9, 2), 5);
        assert_eq!(get_elements_per_block(1, 9, 2), 4);

        assert_eq!(get_elements_per_block(0, 8, 3), 3);
        assert_eq!(get_elements_per_block(1, 8, 3), 3);
        assert_eq!(get_elements_per_block(2, 8, 3), 2);

        assert_eq!(get_elements_per_block(0, 9, 3), 3);
        assert_eq!(get_elements_per_block(1, 9, 3), 3);
        assert_eq!(get_elements_per_block(2, 9, 3), 3);

        assert_eq!(get_elements_per_block(0, 10, 3), 4);
        assert_eq!(get_elements_per_block(1, 10, 3), 3);
        assert_eq!(get_elements_per_block(2, 10, 3), 3);
    }

    #[test]
    fn test_apply_reduced_capacity() {
        let mut data = vec2d![[1, 2], [1, 2], [1, 2]];
        apply_reduced_capacity(&mut data, 2);
        assert_eq!(
            data,
            vec2d![[1, 2], [1, 2], [1, 2]]
        );
        let mut data = vec2d![[1, 2, 3], [1, 2, 3], [1, 2, 3]];
        apply_reduced_capacity(&mut data, 2);
        assert_eq!(
            data,
            vec2d![[1, 2], [3], [1, 2], [3], [1, 2], [3]],
        );
        let mut data = vec2d![[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]];
        apply_reduced_capacity(&mut data, 2);
        assert_eq!(
            data,
            vec2d![[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]],
        );
        let mut data = vec2d![[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]];
        apply_reduced_capacity(&mut data, 2);
        assert_eq!(
            data,
            vec2d![[1, 2], [3, 4], [5], [1, 2], [3, 4], [5], [1, 2], [3, 4], [5]],
        );
    }

    #[test]
    fn test_apply_reduced_capacity_favor_equal_splits() {
        let mut data = vec2d![[1, 2, 3, 4, 5]];
        apply_reduced_capacity(&mut data, 4);
        assert_eq!(
            data,
            vec2d![[1, 2, 3], [4, 5]],
        );
        let mut data = vec2d![[1, 2, 3, 4, 5, 6, 7, 8, 9]];
        apply_reduced_capacity(&mut data, 4);
        assert_eq!(
            data,
            vec2d![[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        );
    }

    #[test]
    fn test_apply_reduced_capacity_smaller_blocks_are_kept() {
        let mut data = vec2d![[1, 2, 3, 4], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]];
        apply_reduced_capacity(&mut data, 3);
        assert_eq!(
            data,
            vec2d![[1, 2], [3, 4], [1], [1, 2], [1, 2, 3], [1, 2], [3, 4]]
        );
    }

    #[test]
    fn test_apply_reduced_capacity_multi_split() {
        let mut data = vec2d![[1, 2, 3, 4, 5, 6]];
        apply_reduced_capacity(&mut data, 2);
        assert_eq!(
            data,
            vec2d![[1, 2], [3, 4], [5, 6]],
        );
        let mut data = vec2d![[1, 2, 3, 4, 5, 6, 7]];
        apply_reduced_capacity(&mut data, 3);
        assert_eq!(
            data,
            vec2d![[1, 2, 3], [4, 5], [6, 7]],
        );
    }

    // ------------------------------------------------------------------------
    // Index handling
    // ------------------------------------------------------------------------

    #[test]
    fn test_get_by_index() {
        let a = new_array!(2, vec![vec![1], vec![2, 3]]);
        assert_eq!(*a.get_by_index(Index::new(0, 0)), 1);
        assert_eq!(*a.get_by_index(Index::new(1, 0)), 2);
        assert_eq!(*a.get_by_index(Index::new(1, 1)), 3);
    }

    #[test]
    fn test_next_index() {
        let a = new_array!(2, vec2d![[1], [2, 3], [4, 5, 6]]);
        assert_eq!(a.next_index(Index::new(0, 0)), Some(Index::new(1, 0)));
        assert_eq!(a.next_index(Index::new(1, 0)), Some(Index::new(1, 1)));
        assert_eq!(a.next_index(Index::new(1, 1)), Some(Index::new(2, 0)));
        assert_eq!(a.next_index(Index::new(2, 0)), Some(Index::new(2, 1)));
        assert_eq!(a.next_index(Index::new(2, 1)), Some(Index::new(2, 2)));
        assert_eq!(a.next_index(Index::new(2, 2)), None);
    }

    #[test]
    fn test_prev_index() {
        let a = new_array!(2, vec2d![[1], [2, 3], [4, 5, 6]]);
        assert_eq!(a.prev_index(Index::new(0, 0)), None);
        assert_eq!(a.prev_index(Index::new(1, 0)), Some(Index::new(0, 0)));
        assert_eq!(a.prev_index(Index::new(1, 1)), Some(Index::new(1, 0)));
        assert_eq!(a.prev_index(Index::new(2, 0)), Some(Index::new(1, 1)));
        assert_eq!(a.prev_index(Index::new(2, 1)), Some(Index::new(2, 0)));
        assert_eq!(a.prev_index(Index::new(2, 2)), Some(Index::new(2, 1)));
    }

    // ------------------------------------------------------------------------
    // Rank (TODO) / min / max
    // ------------------------------------------------------------------------

    #[test]
    fn test_min_max() {
        let a = ArrayStump::new(int_comparator);
        assert_eq!(a.min(), None);
        assert_eq!(a.max(), None);
        let a = new_array!(2, vec2d![[1]]);
        assert_eq!(a.min(), Some(&1));
        assert_eq!(a.max(), Some(&1));
        let a = new_array!(2, vec2d![[1], [2, 3], [4]]);
        assert_eq!(a.min(), Some(&1));
        assert_eq!(a.max(), Some(&4));
        let a = new_array!(2, vec2d![[1, 2], [3, 4]]);
        assert_eq!(a.min(), Some(&1));
        assert_eq!(a.max(), Some(&4));
    }

    #[test]
    fn test_fix_rank_range() {
        let mut a = new_array!(2, vec![vec![2, 4], vec![6, 8], vec![10, 12]]);

        // Wiggling without changes shan't do anything
        let mut c = Some(Index::FIRST);
        while c.is_some() {
            assert_eq!(a.fix_rank_range(c.unwrap(), c.unwrap()).len(), 0);
            c = a.next_index(c.unwrap());
        }
        assert_eq!(a.fix_rank_range(Index::FIRST, Index::new(2,1)).len(), 0);

        // find the right place for an element to go
        a.data[0][0] = 5;
        assert_eq!(a.data, [vec![5, 4], vec![6, 8], vec![10, 12]]);
        a.sort_element_back(Index::FIRST, Index::FIRST);
        assert_eq!(a.data, [vec![4, 5], vec![6, 8], vec![10, 12]]);

        // skipped elements should stay out of order
        a.data[0][1] = 1;
        a.data[1][0] = 3;
        a.sort_element_forward(Index::new(1, 0), Index::new(0, 1));
        assert_eq!(a.data, [vec![3, 4], vec![1, 8], vec![10, 12]]);

        // fix multiple element range at once
        a.data[1][1] = 2;
        a.data[2][0] = 20;
        let ts = a.fix_rank_range(Index::new(1,0),Index::new(2,0));
        assert_eq!(a.data, [vec![1, 2], vec![3, 4], vec![12, 20]]);
        assert_eq!(a.fix_index(&ts, Index::new(1, 0)), Index::new(0, 0));
        assert_eq!(a.fix_index(&ts, Index::new(1, 1)), Index::new(0, 1));
        assert_eq!(a.fix_index(&ts, Index::new(2, 1)), Index::new(2, 0));

        a.data[1][1] = 0;
        let ts = a.fix_rank_range(Index::new(1, 1), Index::new(1, 1));
        assert_eq!(a.fix_index(&ts, Index::new(1, 0)), Index::new(1, 1));

        a.data[0][0] = 4;
        let _ = a.fix_rank_range(Index::FIRST, Index::FIRST);
        assert_eq!(a.data, [vec![1, 2], vec![3, 4], vec![12, 20]]);

        // special case at the end
        a.data[2][0] = 23;
        let _ = a.fix_rank_range(Index::new(2, 1), Index::new(2, 2));
        assert_eq!(a.data, [vec![1, 2], vec![3, 4], vec![20, 23]]);

        // We must never increase capacities while wiggling
        assert_eq!(a.data[0].capacity(), 2);
        assert_eq!(a.data[1].capacity(), 2);
        assert_eq!(a.data[2].capacity(), 2);
    }
}