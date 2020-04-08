extern crate array_stump_benchmarks;

use array_stump_benchmarks::benchmarks::{BenchmarkMode, GeneratorMode, run_benchmarks};

fn main() {
    run_benchmarks(
        BenchmarkMode::Insert{ measure_every: 25},
        GeneratorMode::Asc,
        1000000,
        3,
        false,
    );
}