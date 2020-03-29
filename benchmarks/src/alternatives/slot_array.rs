use std::cmp::Ordering;
use std::iter;

pub struct SlotArray<T, C>
where
    C: Fn(&T, &T) -> Ordering,
{
    spacing: usize,
    comparator: C,
    data_raw: Vec<Option<T>>,
    num_elements: usize,
}

#[allow(dead_code)]
impl<T, C> SlotArray<T, C>
where
    C: Fn(&T, &T) -> Ordering,
    T: Clone + std::fmt::Debug,
{
    pub fn new(comparator: C, initial_capacity: usize, spacing: usize) -> SlotArray<T, C> {
        SlotArray {
            spacing,
            comparator,
            data_raw: iter::repeat(None).take(initial_capacity * (spacing + 1)).collect(),
            num_elements: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.num_elements
    }

    pub fn insert(&mut self, t: T) -> bool {
        let (index_larger_or_equal, equals) = binary_search_by(&self.data_raw, #[inline(always)] |x| (self.comparator)(x, &t));

        if !equals {
            let insert_slot = determine_insert_slot(&self.data_raw, index_larger_or_equal);

            if let Some(idx) = insert_slot {
                self.data_raw[idx] = Some(t);
            } else {
                self.data_raw = redistribute(&self.data_raw, self.num_elements, self.spacing, index_larger_or_equal, t);
            }
            self.num_elements += 1;

            true
        } else {
            false
        }
    }

    pub fn remove(&mut self, t: &T) -> bool {
        let (index_larger_or_equal, equals) = binary_search_by(&self.data_raw, #[inline(always)] |x| (self.comparator)(x, &t));
        if equals {
            self.data_raw[index_larger_or_equal] = None;
            self.num_elements -= 1;
            true
        } else {
            false
        }
    }

    pub fn collect(&self) -> Vec<T> {
        collect(&self.data_raw)
    }

    pub fn debug(&self) {
        println!("{:?}", self.data_raw);
    }

}


pub fn binary_search_by<T, F>(data: &[Option<T>], mut f: F) -> (usize, bool)
where
    F: FnMut(&T) -> Ordering,
    T: std::fmt::Debug,
{
    if data.len() == 0 {
        //return BinarySearchResult::Err;
        return (data.len(), false);
    }
    let mut l: usize = 0;
    let mut r: usize = data.len();
    let mut equals = false;

    while r > l {
        let mid = l + (r - l) / 2;

        // Search around `mid` for an element that is not None
        let mid_el = next(&data, mid, r - 1).or_else(|| prev(&data, mid, l));
        // println!("{} {} {} {:?}", l, r, mid, mid_el);

        if let Some((mid, el)) = mid_el {
            let cmp = f(el);
            match cmp {
                Ordering::Greater => {
                    r = mid;
                }
                Ordering::Equal => {
                    r = mid;
                    equals = true;
                }
                Ordering::Less => {
                    l = mid + 1;
                }
            }
        } else {
            break;
        }
    }

    (r, equals)
}

#[inline]
fn next<'a, T>(data: &'a [Option<T>], idx: usize, bound: usize) -> Option<(usize, &'a T)> {
    let mut i = idx;
    // println!("next {} {}", idx, bound);
    //if idx > bound {
    //    return None;
    //}
    debug_assert!(idx <= bound);
    loop {
        if let Some(el) = unsafe { &data.get_unchecked(i) } {
            return Some((i, el));
        }
        if i == bound {
            return None;
        } else {
            i += 1;
        }
    }
}

#[inline]
fn prev<'a, T>(data: &'a [Option<T>], idx: usize, bound: usize) -> Option<(usize, &'a T)> {
    let mut i = idx;
    // println!("prev {} {}", idx, bound);
    //if idx < bound {
    //    return None;
    //}
    debug_assert!(idx >= bound);
    loop {
        if let Some(el) = unsafe { &data.get_unchecked(i) } {
            return Some((i, el));
        }
        if i == bound {
            return None;
        } else {
            i -= 1;
        }
    }
}

#[inline]
fn determine_insert_slot<'a, T>(data: &'a [Option<T>], insert_index: usize) -> Option<usize> {
    let idx_start = insert_index as i64 - 1;
    let mut idx_low: i64 = idx_start;
    let mut idx_mid: i64 = idx_low;

    loop {
        if idx_low >= 0 && data[idx_low as usize].is_none() {
            idx_low -= 1;
        } else {
            break;
        }
        if idx_low >= 0 && data[idx_low as usize].is_none() {
            idx_low -= 1;
            idx_mid -= 1;
        } else {
            break;
        }
    }

    // println!("{} {}", idx_low, data[idx_low as usize].is_some());
    // println!("{} {}", idx_low, idx_mid);
    debug_assert!(idx_low == -1 || data[idx_low as usize].is_some());

    if idx_low == idx_start {
        None
    } else {
        Some(idx_mid as usize)
    }
}

fn redistribute<'a, T>(data: &'a [Option<T>], num_elements: usize, spacing: usize, insert_index: usize, t: T) -> Vec<Option<T>>
where
    T: Clone
{
    // println!("\nredistribute:");
    if num_elements == 0 {
        let mut new_data: Vec<Option<T>> = iter::repeat(None).take(spacing * 2 + 1).collect();
        new_data[spacing] = Some(t);
        return new_data;
    }

    let new_num_elements = num_elements + 1;
    let new_size = (new_num_elements + 1) * spacing + new_num_elements;
    let mut new_data: Vec<Option<T>> = iter::repeat(None).take(new_size).collect();

    /*
    let first = next(data, 0, data.len());
    if first.is_none() {
        return iter::repeat(None).take(spacing * 2 + 1).collect();
    }

    let mut idx_i = first.unwrap().0;
    let mut idx_o = spacing;

    while idx_i < insert_index {
        new_data[idx_o] = data[idx_i].clone();

        if idx_i + 1 < data.len() {
            let nxt = next(data, idx_i + 1, data.len());
            if let Some((idx_next, _)) = nxt {
                idx_i = idx_next;
                idx_o += spacing + 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    new_data[idx_o] = Some(t);
    idx_o += spacing + 1;
    while idx_i < data.len() {
        new_data[idx_o] = data[idx_i].clone();

        if idx_i + 1 < data.len() {
            let nxt = next(data, idx_i + 1, data.len());
            if let Some((idx_next, _)) = nxt {
                idx_i = idx_next;
                idx_o += spacing + 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    */

    let mut idx_o = spacing;
    let mut inserted = false;

    traverse(data, |idx_i, x| {
        if !inserted && idx_i >= insert_index {
            // println!("insert new element at index {}", idx_o);
            new_data[idx_o] = Some(t.clone());
            idx_o += spacing + 1;
            inserted = true;
        }
        // println!("insert at index {} from {}", idx_o, idx_i);
        new_data[idx_o] = Some(x.clone());
        idx_o += spacing + 1;
    });

    if !inserted {
        new_data[idx_o] = Some(t.clone());
    }

    new_data
}

fn traverse<T, F>(data: &[Option<T>], mut f: F)
where
    T: Clone,
    F: FnMut(usize, &T),
{
    for i in 0 .. data.len() {
        if let Some(x) = &data[i] {
            f(i, x);
        }
    }
}

fn collect<T>(data: &[Option<T>]) -> Vec<T>
where
    T: Clone,
{
    let mut v = Vec::new();
    traverse(data, |_, x: &T| v.push(x.clone()));
    v
}

#[cfg(test)]
mod test {
    use super::*;
    use std::cmp::Ordering;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    pub fn binary_search_by_reference<T, F>(data: &[Option<T>], mut f: F) -> (usize, bool)
    where
        F: FnMut(&T) -> Ordering,
    {
        for i in 0 .. data.len() {
            if let Some(x) = &data[i] {
                let cmp = f(x);
                match cmp {
                    Ordering::Equal => return (i, true),
                    Ordering::Greater => return (i, false),
                    _ => {}
                }
            }
        }
        (data.len(), false)
    }

    fn int_comparator(a: &i32, b: &i32) -> Ordering {
        a.cmp(b)
    }

    #[test]
    fn test_binary_search_by() {
        let data = [Some(1), Some(2), Some(3)];
        assert_eq!(binary_search_by(&data, |x| int_comparator(x, &0)), (0, false));
        assert_eq!(binary_search_by(&data, |x| int_comparator(x, &1)), (0, true));
        assert_eq!(binary_search_by(&data, |x| int_comparator(x, &2)), (1, true));
        assert_eq!(binary_search_by(&data, |x| int_comparator(x, &3)), (2, true));
        assert_eq!(binary_search_by(&data, |x| int_comparator(x, &4)), (3, false));
    }

    fn generate_random_array(rng: &mut StdRng, len: usize) -> (Vec<Option<i32>>, Vec<i32>) {
        let mut data = Vec::new();
        let mut values = Vec::new();

        let mut last = 0;
        for _ in 0 .. len {
            data.push(Some(last));
            values.push(last);
            if rng.gen::<bool>() {
                last += 1;
            }
        }

        if len > 0 {
            let min = values.iter().min().unwrap() - 1;
            let max = values.iter().max().unwrap() + 1;
            values.extend(&[min, max]);
        }
        (data, values)
    }

    fn insert_random_slots(rng: &mut StdRng, data: &[Option<i32>], slot_prob: f64) -> Vec<Option<i32>> {
        let mut data_with_slots = Vec::new();
        let mut i = 0;
        while i < data.len() {
            if rng.gen_range(0.0, 1.0) < slot_prob {
                data_with_slots.push(None);
            } else {
                data_with_slots.push(data[i]);
                i += 1;
            }
        }
        data_with_slots
    }

    fn test_against_reference(data: &[Option<i32>], value: i32) {
        println!("{:?} {}", data, value);

        let result_actual = binary_search_by(&data, |x| x.cmp(&value));
        let result_expect = binary_search_by_reference(&data, |x| x.cmp(&value));
        assert_eq!(result_actual, result_expect);
    }

    #[test]
    fn test_binary_search_by_random() {
        let mut rng: StdRng = SeedableRng::seed_from_u64(0);
        for array_len in 0 .. 16 {
            for _ in 0 .. 10 {
                let (data, values) = generate_random_array(&mut rng, array_len);

                let data_variations = vec![
                    data.clone(),
                    insert_random_slots(&mut rng, &data, 0.1),
                    insert_random_slots(&mut rng, &data, 0.1),
                    insert_random_slots(&mut rng, &data, 0.5),
                    insert_random_slots(&mut rng, &data, 0.9),
                    insert_random_slots(&mut rng, &data, 0.999),
                ];

                for data in &data_variations {
                    for value in &values {
                        test_against_reference(data, *value);
                    }
                }
            }
        }
    }

    #[test]
    fn test_determine_insert_slot() {
        // cases without free slot
        assert_eq!(
            determine_insert_slot(&[Some(0)], 0), None
        );
        assert_eq!(
            determine_insert_slot(&[Some(0)], 1), None
        );
        assert_eq!(
            determine_insert_slot(&[Some(0), Some(1)], 0), None
        );
        assert_eq!(
            determine_insert_slot(&[Some(0), Some(1)], 1), None
        );
        assert_eq!(
            determine_insert_slot(&[Some(0), Some(1)], 2), None
        );

        assert_eq!(
            determine_insert_slot(&[None, Some(0)], 1), Some(0)
        );
        assert_eq!(
            determine_insert_slot(&[None, None, Some(0)], 2), Some(0)
        );
        assert_eq!(
            determine_insert_slot(&[None, None, None, Some(0)], 3), Some(1)
        );
        assert_eq!(
            determine_insert_slot(&[None, None, None, None, Some(0)], 4), Some(1)
        );
        assert_eq!(
            determine_insert_slot(&[None, None, None, None, None, Some(0)], 5), Some(2)
        );

        assert_eq!(
            determine_insert_slot(&[Some(0), None, Some(1)], 2), Some(1)
        );
        assert_eq!(
            determine_insert_slot(&[Some(0), None, None, Some(1)], 3), Some(1)
        );
        assert_eq!(
            determine_insert_slot(&[Some(0), None, None, None, Some(1)], 4), Some(2)
        );

        assert_eq!(
            determine_insert_slot(&[Some(0), None], 2), Some(1)
        );
        assert_eq!(
            determine_insert_slot(&[Some(0), None, None], 3), Some(1)
        );
        assert_eq!(
            determine_insert_slot(&[Some(0), None, None, None], 4), Some(2)
        );

        let all_none: &[Option<i32>] = &[None, None, None];
        assert_eq!(determine_insert_slot(all_none, 0), None);
        assert_eq!(determine_insert_slot(all_none, 1), Some(0));
        assert_eq!(determine_insert_slot(all_none, 2), Some(0));
        assert_eq!(determine_insert_slot(all_none, 3), Some(1));
    }

    #[test]
    fn test_traverse() {
        let v_empty1: Vec<Option<i32>> = vec![];
        let v_empty2: Vec<Option<i32>> = vec![None];
        assert_eq!(collect(&v_empty1), Vec::<i32>::new());
        assert_eq!(collect(&v_empty2), Vec::<i32>::new());
        assert_eq!(collect(&[Some(20), Some(30)]), vec![20, 30]);
        assert_eq!(collect(&[None, Some(20), None, Some(30), None]), vec![20, 30]);
    }

    //#[ignore]
    #[test]
    fn test_redistribute() {
        assert_eq!(
            redistribute(&[Some(20), Some(30)], 2, 0, 0, 10),
            vec![Some(10), Some(20), Some(30)]
        );
        assert_eq!(
            redistribute(&[Some(20), Some(30)], 2, 0, 1, 25),
            vec![Some(20), Some(25), Some(30)]
        );
        assert_eq!(
            redistribute(&[Some(20), Some(30)], 2, 0, 2, 40),
            vec![Some(20), Some(30), Some(40)]
        );

        assert_eq!(
            redistribute(&[Some(20), Some(30)], 2, 2, 0, 10),
            vec![None, None, Some(10), None, None, Some(20), None, None, Some(30), None, None]
        );
        assert_eq!(
            redistribute(&[Some(20), Some(30)], 2, 2, 1, 25),
            vec![None, None, Some(20), None, None, Some(25), None, None, Some(30), None, None]
        );
        assert_eq!(
            redistribute(&[Some(20), Some(30)], 2, 2, 2, 40),
            vec![None, None, Some(20), None, None, Some(30), None, None, Some(40), None, None]
        );

        let v_empty: Vec<Option<i32>> = vec![];
        assert_eq!(
            redistribute(&v_empty, 0, 0, 0, 42),
            vec![Some(42)]
        );
        assert_eq!(
            redistribute(&v_empty, 0, 1, 0, 42),
            vec![None, Some(42), None]
        );

    }
}

/*
impl<T, C> SplaySet<T, C>
where
    C: Fn(&T, &T) -> Ordering,
{
    pub fn new(comparator: C) -> SplaySet<T, C> {
        SplaySet {
            tree: SplayTree::new(comparator),
        }
    }

    pub fn len(&self) -> usize {
        self.tree.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn clear(&mut self) {
        self.tree.clear()
    }

    pub fn contains(&self, t: &T) -> bool {
        self.tree.contains(t)
    }

    pub fn find(&self, t: &T) -> Option<&T> {
        self.tree.find_key(t)
    }

    pub fn next(&self, t: &T) -> Option<&T> {
        self.tree.next(t).map(|kv| kv.0)
    }

    pub fn prev(&self, t: &T) -> Option<&T> {
        self.tree.prev(t).map(|kv| kv.0)
    }

    pub fn insert(&mut self, t: T) -> bool {
        self.tree.insert(t, ()).is_none()
    }

    pub fn remove(&mut self, t: &T) -> bool {
        self.tree.remove(t).is_some()
    }

    pub fn min(&self) -> Option<&T> {
        self.tree.min()
    }

    pub fn max(&self) -> Option<&T> {
        self.tree.max()
    }

    pub fn traverse<F>(&self, traverse: &mut F) where F: FnMut(&T) {
        self.tree.traverse(&mut |k, _| traverse(k));
    }
}

impl<T, C> IntoIterator for SplaySet<T, C>
where
    C: Fn(&T, &T) -> Ordering,
{
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.tree.into_iter(),
        }
    }
}

pub struct IntoIter<T> {
    inner: tree::IntoIter<T, ()>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.inner.next().map(|p| p.0)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<T> {
        self.inner.next_back().map(|(k, _)| k)
    }
}

impl<T, C> Extend<T> for SplaySet<T, C>
where
    C: Fn(&T, &T) -> Ordering,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, i: I) {
        for t in i {
            self.insert(t);
        }
    }
}

*/