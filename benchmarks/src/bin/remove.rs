extern crate array_stump_benchmarks;

use array_stump_benchmarks::benchmarks::{BenchmarkMode, GeneratorMode, run_benchmarks};

fn main() {
    run_benchmarks(
        BenchmarkMode::Remove{ measure_every: 25},
        GeneratorMode::Avg,
        100000,
        3,
        false,
    );
}