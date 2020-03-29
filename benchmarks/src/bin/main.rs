extern crate array_stump_benchmarks;
extern crate array_stump;

use array_stump_benchmarks::create_cmp;
use array_stump_benchmarks::helpers;
use array_stump_benchmarks::helpers::{FloatWrapper, get_num_calls_b_tree};

use array_stump::ArrayStump;
use std::collections::BTreeSet;
use array_stump_benchmarks::alternatives::splay::SplaySet;
use array_stump_benchmarks::alternatives::slot_array::SlotArray;
use array_stump_benchmarks::alternatives::plain_array::PlainArray;

use std::time::Instant;
use pretty_assertions::assert_eq;


create_cmp!(cmp_array_stump, get_num_calls_array_stump, NUM_CALLS_ARRAY_STUMP);
create_cmp!(cmp_splay_tree, get_num_calls_splay_tree, NUM_CALLS_SPLAY_TREE);
create_cmp!(cmp_slot_array, get_num_calls_slot_array, NUM_CALLS_SLOT_ARRAY);
create_cmp!(cmp_plain_array, get_num_calls_plain_array, NUM_CALLS_PLAIN_ARRAY);

/*
TODO: Check if it is possible to make the following work somehow:

type Compare = Fn(&f64, &f64) -> std::cmp::Ordering;

trait GenericBenchmark<T> {
    type T;
    fn init() -> T;
    fn insert(t: T) -> bool;
}

struct ArrayStumpBenchmark;

impl<C> GenericBenchmark<ArrayStump<f64, C>> for ArrayStumpBenchmark
where
    C: Fn(&f64, &f64) -> std::cmp::Ordering
{
    type T = ArrayStump<f64, C>;

    fn init() -> Self::T {
        ArrayStump::new(cmp_array_tree, 512)
    }
    fn insert(t: &mut Self::T) -> bool {
        true
    }
}
*/

#[derive(Clone, Copy)]
enum BenchmarkMode {
    Fill { measure_every: usize },
}

fn run_generic_benchmark<T, F1, F2, F3>(mode: BenchmarkMode, values: &[f64], init: F1, insert: F2, get_len: F3) -> Vec<(usize, f64)>
where
    F1: Fn() -> T,
    F2: Fn(&mut T, f64),
    F3: Fn(&T) -> usize,

{
    let mut set = init();
    let mut elapsed_times = Vec::with_capacity(values.len());

    match mode {
        BenchmarkMode::Fill{ measure_every } => {
            let start = Instant::now();
            for (i, x) in values.iter().enumerate() {
                insert(&mut set, *x);

                let len = i + 1;
                if len % measure_every == 0 {
                    elapsed_times.push((len, start.elapsed().as_secs_f64()));
                }
            }
            assert_eq!(get_len(&set), values.len());
        }
    }

    elapsed_times
}

use std::rc::Rc;

type BenchFunc = Rc<dyn Fn(BenchmarkMode, &[f64]) -> Vec<(usize, f64)>>;

struct AllBenches {
    bench_array_stump: BenchFunc,
    bench_splay_tree: BenchFunc,
    bench_b_tree: BenchFunc,
    bench_slot_array: BenchFunc,
    bench_plain_array: BenchFunc,
}

impl AllBenches {
    fn new() -> Self {
        let bench_array_stump = |mode: BenchmarkMode, values: &[f64]| {
            run_generic_benchmark(
                mode,
                &values,
                || ArrayStump::new(cmp_array_stump, 512),
                |set, x| { set.insert(x); },
                |set| set.len(),
            )
        };
        let bench_splay_tree = |mode: BenchmarkMode, values: &[f64]| {
            run_generic_benchmark(
                mode,
                &values,
                || SplaySet::new(cmp_splay_tree),
                |set, x| { set.insert(x); },
                |set| set.len(),
            )
        };
        let bench_b_tree = |mode: BenchmarkMode, values: &[f64]| {
            run_generic_benchmark(
                mode,
                &values,
                || BTreeSet::new(),
                |set, x| { set.insert(FloatWrapper(x)); },
                |set| set.len(),
            )
        };
        let bench_slot_array = |mode: BenchmarkMode, values: &[f64]| {
            run_generic_benchmark(
                mode,
                &values,
                || SlotArray::new(cmp_slot_array, 20, 4),
                |set, x| { set.insert(x); },
                |set| set.len(),
            )
        };
        let bench_plain_array = |mode: BenchmarkMode, values: &[f64]| {
            run_generic_benchmark(
                mode,
                &values,
                || PlainArray::new(cmp_plain_array, 1024),
                |set, x| { set.insert(x); },
                |set| set.len(),
            )
        };
        AllBenches {
            bench_array_stump: Rc::new(bench_array_stump),
            bench_splay_tree: Rc::new(bench_splay_tree),
            bench_b_tree: Rc::new(bench_b_tree),
            bench_slot_array: Rc::new(bench_slot_array),
            bench_plain_array: Rc::new(bench_plain_array),
        }
    }
}

#[derive(Clone)]
struct BenchmarkTask
{
    name: String,
    func: BenchFunc,
    run: i32,
}

fn construct_benchmark_tasks(all_benches: &AllBenches, run: i32, all_combatants: bool) -> Vec<BenchmarkTask> {
    let mut benchmarks: Vec<BenchmarkTask> = vec![
        BenchmarkTask {
            run,
            name: "ArrayStump".to_string(),
            func: all_benches.bench_array_stump.clone(),
        },
        BenchmarkTask {
            run,
            name: "SplayTree".to_string(),
            func: all_benches.bench_splay_tree.clone(),
        },
        BenchmarkTask {
            run,
            name: "BTree".to_string(),
            func: all_benches.bench_b_tree.clone(),
        },
    ];
    if all_combatants {
        benchmarks.extend(vec![
            BenchmarkTask {
                run,
                name: "SlotArray".to_string(),
                func: all_benches.bench_slot_array.clone(),
            },
            BenchmarkTask {
                run,
                name: "PlainArray".to_string(),
                func: all_benches.bench_plain_array.clone(),
            },
        ])
    }
    helpers::shuffle(&mut benchmarks);
    benchmarks
}


fn run_fill_benchmarks() {
    let n = 1000000;
    let num_runs = 3;
    let all_combatants = false;

    let mode = BenchmarkMode::Fill{ measure_every: 25};

    let all_benches = AllBenches::new();

    for run in 0..=num_runs {
        let benchmark_tasks = construct_benchmark_tasks(&all_benches, run, all_combatants);

        let values = helpers::gen_rand_values(n);
        assert_eq!(values.len(), n);

        for benchmark_task in benchmark_tasks {
            println!("Running benchmark task: {} / {}", benchmark_task.name, benchmark_task.run);

            let measurements = (benchmark_task.func)(mode, &values);

            // Use zero-th iteration for warm up
            if run > 0 {
                let iters: Vec<_> = measurements.iter().map(|i_t| i_t.0).collect();
                let times: Vec<_> = measurements.iter().map(|i_t| i_t.1).collect();
                helpers::export_elapsed_times(
                    &benchmark_task.name,
                    benchmark_task.run,
                    &format!("results/fill_avg_{}_{}.json", benchmark_task.name, benchmark_task.run),
                    &iters,
                    &times,
                );
            }
        }
    }

    helpers::call_plots();
}


fn main() {
    if cfg!(debug_assertions) {
        println!("WARNING: Debug assertions are enabled. Benchmarking should be done in `--release`.");
    }

    run_fill_benchmarks();

    println!("Num calls array stump: {:12}", get_num_calls_array_stump());
    println!("Num calls splay tree:  {:12}", get_num_calls_splay_tree());
    println!("Num calls B tree:      {:12}", get_num_calls_b_tree());
    println!("Num calls slot array:  {:12}", get_num_calls_slot_array());
    println!("Num calls plain array: {:12}", get_num_calls_plain_array());
}