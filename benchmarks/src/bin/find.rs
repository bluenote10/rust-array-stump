extern crate array_stump_benchmarks;

use array_stump_benchmarks::benchmarks::{BenchmarkMode, run_benchmarks};

fn main() {
    run_benchmarks(
        BenchmarkMode::Find{ measure_every: 100},
        1000000,
        3,
        false,
    );
}