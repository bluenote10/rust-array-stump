//extern crate array_stump;
use std::cmp::Ordering;

use super::helpers::gen_rand_values_i32;

use array_stump::ArrayStump;
use super::alternatives::splay::SplaySet;


fn cmp(a: &i32, b: &i32) -> Ordering {
    a.cmp(b)
}


#[test]
fn fill() {
    for array_len in 0 .. 64 {
        for cap in 2 .. 64 {
            let values = gen_rand_values_i32(array_len);
            // println!("\nInserting: {:?}", values);

            let mut set_a = ArrayStump::new(cmp, cap);
            let mut set_b = SplaySet::new(cmp);

            for x in &values {
                let res_a = set_a.insert(*x);
                let res_b = set_b.insert(*x);
                // println!("{} {} {}", x, res_a, res_b);
                assert_eq!(res_a, res_b);
            }

            let data_a = set_a.collect();
            let data_b: Vec<_> = set_b.into_iter().collect();

            assert_eq!(data_a.len(), data_b.len());
            assert_eq!(data_a, data_b);
        }
    }
}
