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


create_cmp!(cmp_array_tree, get_num_calls_array_tree, NUM_CALLS_ARRAY_TREE);
create_cmp!(cmp_splay_tree, get_num_calls_splay_tree, NUM_CALLS_SPLAY_TREE);
create_cmp!(cmp_slot_array, get_num_calls_slot_array, NUM_CALLS_SLOT_ARRAY);
create_cmp!(cmp_plain_array, get_num_calls_plain_array, NUM_CALLS_PLAIN_ARRAY);


fn generic_fill_benchmark<T, F1, F2, F3>(values: &[f64], measure_every: i32, init: F1, insert: F2, get_len: F3) -> Vec<(usize, f64)>
where
    F1: Fn() -> T,
    F2: Fn(&mut T, f64),
    F3: Fn(&T) -> usize,

{
    let mut set = init();

    let mut elapsed_times = Vec::with_capacity(values.len());

    let start = Instant::now();
    for (i, x) in values.iter().enumerate() {
        insert(&mut set, *x);

        let len = i + 1;
        if len % measure_every as usize == 0 {
            elapsed_times.push((len, start.elapsed().as_secs_f64()));
        }
    }
    assert_eq!(get_len(&set), values.len());

    elapsed_times

}

/*
fn benchmark_fill_array_tree(values: &[f64]) -> usize {
    let mut set = ArrayTree::new(cmp_array_tree, 16);
    for x in values {
        set.insert(*x);
    }
    set.len()
}

fn benchmark_fill_splay_tree(values: &[f64]) -> usize {
    let mut set = SplaySet::new(cmp_splay_tree);
    for x in values {
        set.insert(*x);
    }
    set.len()
}

fn benchmark_fill_b_tree(values: &[f64]) -> usize {
    let mut set = BTreeSet::new();
    for x in values {
        set.insert(FloatWrapper(*x));
    }
    set.len()
}
*/

/*
struct Benchmark<F>
where
    F: Fn(&[f64]) -> usize
{
    name: String,
    func: F,
}
*/
#[derive(Clone)]
struct Benchmark<'a>
{
    name: String,
    //func: fn(&[f64]) -> usize,
    func: &'a dyn Fn(&[f64]) -> Vec<(usize, f64)>,
    run: i32,
}


fn run_fill_benchmarks() {

    let n = 1000000;
    let measure_every = 25;
    let num_runs = 3;
    let all_combatants = false;

    let fill_array_tree = |values: &[f64]| {
        generic_fill_benchmark(
            &values,
            measure_every,
            || ArrayStump::new(cmp_array_tree, 512),
            |set, x| { set.insert(x); },
            |set| set.len(),
        )
    };

    let fill_splay_tree = |values: &[f64]| {
        generic_fill_benchmark(
            &values,
            measure_every,
            || SplaySet::new(cmp_splay_tree),
            |set, x| { set.insert(x); },
            |set| set.len(),
        )
    };


    let fill_b_tree = |values: &[f64]| {
        generic_fill_benchmark(
            &values,
            measure_every,
            || BTreeSet::new(),
            |set, x| { set.insert(FloatWrapper(x)); },
            |set| set.len(),
        )
    };

    let fill_slot_array = |values: &[f64]| {
        generic_fill_benchmark(
            &values,
            measure_every,
            || SlotArray::new(cmp_slot_array, 20, 4),
            |set, x| { set.insert(x); },
            |set| set.len(),
        )
    };

    let fill_plain_array = |values: &[f64]| {
        generic_fill_benchmark(
            &values,
            measure_every,
            || PlainArray::new(cmp_plain_array, 1024),
            |set, x| { set.insert(x); },
            |set| set.len(),
        )
    };

    for run in 0..=num_runs {
        let mut benchmarks: Vec<Benchmark> = vec![
            Benchmark {
                run,
                name: "ArrayStump".to_string(),
                func: &fill_array_tree,
            },
            Benchmark {
                run,
                name: "SplayTree".to_string(),
                func: &fill_splay_tree,
            },
            Benchmark {
                run,
                name: "BTree".to_string(),
                func: &fill_b_tree,
            },
        ];
        if all_combatants {
            benchmarks.extend(vec![
                Benchmark {
                    run,
                    name: "SlotArray".to_string(),
                    func: &fill_slot_array,
                },
                Benchmark {
                    run,
                    name: "PlainArray".to_string(),
                    func: &fill_plain_array,
                },
            ])
        }
        let benchmarks = helpers::shuffle(&benchmarks);

        let values = helpers::gen_rand_values(n);
        assert_eq!(values.len(), n);

        for benchmark in benchmarks {
            println!("Running benchmark: {} / {}", benchmark.name, benchmark.run);

            let measurements = (benchmark.func)(&values);

            let iters: Vec<_> = measurements.iter().map(|i_t| i_t.0).collect();
            let times: Vec<_> = measurements.iter().map(|i_t| i_t.1).collect();

            if run > 0 {
                helpers::export_elapsed_times(
                    &benchmark.name,
                    benchmark.run,
                    &format!("results/fill_avg_{}_{}.json", benchmark.name, benchmark.run),
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

    println!("Num calls array stump: {:12}", get_num_calls_array_tree());
    println!("Num calls splay tree:  {:12}", get_num_calls_splay_tree());
    println!("Num calls B tree:      {:12}", get_num_calls_b_tree());
    println!("Num calls slot array:  {:12}", get_num_calls_slot_array());
    println!("Num calls plain array: {:12}", get_num_calls_plain_array());
}