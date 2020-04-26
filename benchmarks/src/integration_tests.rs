//extern crate array_stump;
use std::cmp::Ordering;

use super::helpers::{gen_rand_values_i32, shuffle_clone};

use array_stump::ArrayStump;
use super::alternatives::splay::SplaySet;


// TODO: Similar functionality is in core array_stump. Introduce "testing" feature, if more
// code sharing is needed.
pub fn enrich_with_neighbors(data: &[i32]) -> Vec<i32> {
    let mut test_values = Vec::with_capacity(data.len() * 3);
    for x in data {
        test_values.push(*x - 1);
        test_values.push(*x);
        test_values.push(*x + 1);
    }
    test_values
}


fn cmp(a: &i32, b: &i32) -> Ordering {
    a.cmp(b)
}


#[test]
fn insert_and_remove() {
    let repeats = 8;
    for _ in 0 .. repeats {
        for array_len in 0 .. 64 {
            for cap in 2 .. 64 {
                let values = gen_rand_values_i32(array_len);
                // println!("\nInserting: {:?}", values);

                let mut set_a = ArrayStump::new_explicit(cmp, cap);
                let mut set_b = SplaySet::new(cmp);

                for x in &values {
                    let res_a = set_a.insert(*x);
                    let res_b = set_b.insert(*x);
                    // println!("{} {} {}", x, res_a, res_b);
                    assert_eq!(res_a.is_some(), res_b);
                    assert_eq!(set_a.len(), set_b.len());
                    assert_eq!(set_a.collect(), set_b.collect());

                    // Test for index correctness
                    if let Some(res_a) = res_a {
                        assert_eq!(set_a[res_a], *x);
                    }
                }

                let values = shuffle_clone(&values);
                for x in &values {
                    let res_a = set_a.remove(x);
                    let res_b = set_b.remove(x);
                    // println!("{} {} {}", x, res_a, res_b);
                    assert_eq!(res_a, res_b);
                    assert_eq!(set_a.len(), set_b.len());
                    assert_eq!(set_a.collect(), set_b.collect());
                }

            }
        }
    }
}

#[test]
fn find() {
    let repeats = 8;
    for _ in 0 .. repeats {
        for array_len in 0 .. 16 {
            for cap in 2 .. 16 {
                let values = gen_rand_values_i32(array_len);

                let mut set_a = ArrayStump::new_explicit(cmp, cap);
                let mut set_b = SplaySet::new(cmp);

                for x in &values {
                    set_a.insert(*x);
                    set_b.insert(*x);

                    let existing_values = set_a.collect();
                    let existing_values_enriched = enrich_with_neighbors(&existing_values);

                    for y in existing_values_enriched {
                        let res_a = set_a.find(&y);
                        let res_b = set_b.find(&y);
                        assert_eq!(res_a.is_some(), res_b.is_some());
                    }

                }
            }
        }
    }
}
