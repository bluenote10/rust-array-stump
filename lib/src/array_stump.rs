use std::cmp::Ordering;

// The core data structure representing a two-level sorted stump.
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

    /// Insert a value.
    pub fn insert(&mut self, t: T) -> bool {
        // println!("\nInserting: {:?}", t);
        // println!("{:?}", self.data);
        if self.data.len() == 0 {
            self.data.push(self.new_block(t));
            self.num_elements += 1;
            return true;
        }

        // Binary search for block index
        let (idx_block, equals) = binary_search_by(
            &self.data,
            |block| (self.comparator)(&block[0], &t),
        );
        if equals {
            return false;
        }

        // Convert from "first larger" to "last smaller" index semantics
        let mut idx_block = if idx_block > 0 {
            idx_block - 1
        } else {
            0
        };
        // println!("idx_block: {}    block_len: {}", idx_block, self.data[idx_block].len());

        // Split block if necessary
        if self.data[idx_block].len() >= self.capacity as usize {
            let tail_from = (self.capacity / 2) as usize;
            let tail_upto = self.capacity as usize;
            let block_tail = self.data[idx_block][tail_from .. tail_upto].to_vec();

            // Note: The `.to_vec()` requires T: Clone but is faster than using drain. Keep?
            // let block_tail: Vec<_> = self.data[idx_block].drain(tail_from .. tail_upto).collect();

            self.data[idx_block].truncate(tail_from);
            self.data.insert(idx_block + 1, block_tail);

            // println!("block l: {:?}", self.data[idx_block]);
            // println!("block r: {:?}", self.data[idx_block + 1]);
            // Determine into which of the two split blocks the new value goes.
            let cmp = (self.comparator)(&t, &self.data[idx_block + 1][0]);
            if cmp == Ordering::Equal {
                return false;
            } else if cmp == Ordering::Greater {
                idx_block += 1;
            }
            // println!("idx_block: {}", idx_block);
        }

        // Binary search for value index
        let (idx_value, equals) = binary_search_by(
            &self.data[idx_block],
            |x| (self.comparator)(&x, &t),
        );
        if equals {
            return false;
        }
        // println!("idx_value: {}", idx_value);

        // Value insert
        let block_len = self.data[idx_block].len();
        if idx_value < block_len {
            // println!("block: {:?}", self.data[idx_block]);
            self.data[idx_block].insert(idx_value, t);
            // println!("block: {:?}", self.data[idx_block]);
        } else {
            self.data[idx_block].push(t);
        }

        self.num_elements += 1;

        if self.data.len() > self.capacity as usize * 5 {
            self.capacity *= 2;
        }

        true
    }

    /// Remove a value.
    pub fn remove(&mut self, t: &T) -> bool {
        // println!("\nRemoving: {:?}", t);
        if self.data.len() == 0 {
            return false;
        }

        if let Some((idx_block, idx_value)) = self.find(t) {
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

            true
        } else {
            false
        }
    }

    /// Try to find an existing value.
    #[inline]
    pub fn find(&self, t: &T) -> Option<(usize, usize)> {
        if self.data.len() == 0 {
            return None;
        }

        // Binary search for block index
        let (idx_block, equals) = binary_search_by(
            &self.data,
            #[inline] |block| (self.comparator)(&block[0], &t),
        );
        if equals {
            return Some((idx_block, 0));
        }

        // Convert from "first larger" to "last smaller" index semantics
        let idx_block = if idx_block > 0 {
            idx_block - 1
        } else {
            0
        };
        /*
        if idx_block == self.data.len() {
            return None;
        }
        */

        // Binary search for value index
        let (idx_value, equals) = binary_search_by(
            &self.data[idx_block],
            #[inline] |x| (self.comparator)(&x, &t),
        );
        if equals {
            return Some((idx_block, idx_value));
        }

        None
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
}

// Note: We are using our own implementation of binary search, because the implementation
// in the standard library is optimized for fast comparison functions, and requires more
// comparison function evaluations.
fn binary_search_by<T, F>(data: &[T], mut f: F) -> (usize, bool)
where
    F: FnMut(&T) -> Ordering,
    T: std::fmt::Debug,
{
    if data.len() == 0 {
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

    fn generate_random_array(rng: &mut StdRng, len: usize) -> (Vec<i32>, Vec<i32>) {
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
        assert_eq!(equals, false);  // no insertion
    }

    #[test]
    fn test_array_stump_collect() {
        for cap in vec![2, 3, 4, 5] {
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
        assert_eq!(a.find(&2), Some((0, 0)));
        assert_eq!(a.find(&4), Some((0, 1)));
        assert_eq!(a.find(&6), Some((1, 0)));
        assert_eq!(a.find(&8), Some((2, 0)));
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

}