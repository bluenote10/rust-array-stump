extern crate array_stump_benchmarks;
extern crate array_stump;

use array_stump_benchmarks::create_cmp;
use array_stump_benchmarks::helpers;

use array_stump::ArrayStump;

use std::time::Instant;
use pretty_assertions::assert_eq;

create_cmp!(cmp_array_tree, get_num_calls_array_tree, NUM_CALLS_ARRAY_TREE);

fn run_fill_statistics() {
    let n = 1000000;
    let measure_every = 10;
    let values = helpers::gen_rand_values(n);

    let mut set = ArrayStump::new(cmp_array_tree, 256);

    let mut iters = Vec::new();
    let mut times = Vec::new();
    let mut fill_ratio = Vec::new();
    let mut fill_min = Vec::new();
    let mut fill_max = Vec::new();
    let mut num_blocks = Vec::new();
    let mut capacity = Vec::new();
    let mut num_cmp_calls = Vec::new();


    let start = Instant::now();
    for (i, x) in values.iter().enumerate() {
        set.insert(*x);
        let len = i + 1;
        if len % measure_every == 0 {
            iters.push(len);
            times.push(start.elapsed().as_secs_f64());
            fill_ratio.push(set.get_leaf_fill_ratio());
            fill_min.push(set.get_leaf_fill_min());
            fill_max.push(set.get_leaf_fill_max());
            num_blocks.push(set.get_num_blocks());
            capacity.push(set.get_capacity());
            num_cmp_calls.push(get_num_calls_array_tree());
        }
    }
    assert_eq!(set.len(), values.len());

    let mut values_to_remove = values.to_vec();
    helpers::shuffle(&mut values_to_remove);

    for (i, x) in values_to_remove.iter().enumerate() {
        set.remove(x);
        let len = i + 1;
        if len % measure_every == 0 {
            iters.push(len);
            times.push(start.elapsed().as_secs_f64());
            fill_ratio.push(set.get_leaf_fill_ratio());
            fill_min.push(set.get_leaf_fill_min());
            fill_max.push(set.get_leaf_fill_max());
            num_blocks.push(set.get_num_blocks());
            capacity.push(set.get_capacity());
            num_cmp_calls.push(get_num_calls_array_tree());
        }
    }

    helpers::export_stats(&iters, &times, &fill_ratio, &fill_min, &fill_max, &num_blocks, &capacity, &num_cmp_calls);
    helpers::call_plots_stats();
}

fn main() {
    run_fill_statistics();
}
