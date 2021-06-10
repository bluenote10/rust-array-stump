use super::create_cmp;
use super::helpers;
use super::helpers::{get_num_calls_b_tree, FloatWrapper};

use super::alternatives::plain_array::PlainArray;
use super::alternatives::slot_array::SlotArray;
use super::alternatives::splay::SplaySet;
use array_stump::ArrayStump;
use skiplist::ordered_skiplist::OrderedSkipList;
use std::collections::BTreeSet;

use pretty_assertions::assert_eq;
use std::rc::Rc;
use std::time::Instant;


create_cmp!(
    cmp_array_stump,
    get_num_calls_array_stump,
    NUM_CALLS_ARRAY_STUMP
);
create_cmp!(
    cmp_splay_tree,
    get_num_calls_splay_tree,
    NUM_CALLS_SPLAY_TREE
);
create_cmp!(
    cmp_slot_array,
    get_num_calls_slot_array,
    NUM_CALLS_SLOT_ARRAY
);
create_cmp!(
    cmp_plain_array,
    get_num_calls_plain_array,
    NUM_CALLS_PLAIN_ARRAY
);
create_cmp!(cmp_skip_list, get_num_calls_skip_list, NUM_CALLS_SKIP_LIST);


#[derive(Clone, Copy)]
pub struct BenchmarkParams {
    pub n: usize,
    pub measure_every: usize,
    pub num_runs: i32,
    pub all_combatants: bool,
}

#[derive(Clone, Copy)]
pub enum BenchmarkMode {
    Insert,
    Remove,
    Find { recent: bool },
}

impl std::fmt::Display for BenchmarkMode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let name = match self {
            BenchmarkMode::Insert { .. } => "insert",
            BenchmarkMode::Remove { .. } => "remove",
            BenchmarkMode::Find { recent } => {
                if *recent {
                    "find_recent"
                } else {
                    "find_rand"
                }
            }
        };
        write!(f, "{}", name)
    }
}

#[derive(Clone, Copy)]
pub enum GeneratorMode {
    Avg,
    Asc,
    Dsc,
}

impl std::fmt::Display for GeneratorMode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let name = match self {
            GeneratorMode::Avg => "avg",
            GeneratorMode::Asc => "asc",
            GeneratorMode::Dsc => "dsc",
        };
        write!(f, "{}", name)
    }
}

#[allow(clippy::too_many_arguments)]
fn run_generic_benchmark<T, Init, Insert, Remove, GetLen, Find>(
    mode: BenchmarkMode,
    params: BenchmarkParams,
    values: &[f64],
    init: Init,
    insert: Insert,
    remove: Remove,
    get_len: GetLen,
    find: Find,
) -> Vec<(usize, f64)>
where
    Init: Fn() -> T,
    Insert: Fn(&mut T, f64) -> bool,
    Remove: Fn(&mut T, f64) -> bool,
    GetLen: Fn(&T) -> usize,
    Find: Fn(&T, f64) -> bool,
{
    let mut set = init();
    let mut elapsed_times = Vec::with_capacity(values.len() / params.measure_every);

    match mode {
        BenchmarkMode::Insert => {
            let start = Instant::now();
            for (i, x) in values.iter().enumerate() {
                insert(&mut set, *x);

                let len = i + 1;
                if len % params.measure_every == 0 {
                    elapsed_times.push((len, start.elapsed().as_secs_f64()));
                }
            }
            assert_eq!(get_len(&set), values.len());
        }
        BenchmarkMode::Remove => {
            // Insert
            for x in values {
                insert(&mut set, *x);
            }
            assert_eq!(get_len(&set), values.len());

            let mut values_to_remove = values.to_vec();
            helpers::shuffle(&mut values_to_remove);

            // Remove
            let start = Instant::now();
            for (i, x) in values_to_remove.iter().enumerate() {
                remove(&mut set, *x);

                let len = i + 1;
                if len % params.measure_every == 0 {
                    elapsed_times.push((len, start.elapsed().as_secs_f64()));
                }
            }
            assert_eq!(get_len(&set), 0);

            // Note: we reverse the elapsed times so that the reported N corresponds to the collection size.
            let mut elapsed_times_reversed = Vec::with_capacity(elapsed_times.len());
            let mut t = 0.0;
            let mut n = 0;
            for i in (0 .. elapsed_times.len()).rev() {
                let delta_t = if i > 0 {
                    elapsed_times[i].1 - elapsed_times[i - 1].1
                } else {
                    elapsed_times[0].1
                };
                let delta_n = if i > 0 {
                    elapsed_times[i].0 - elapsed_times[i - 1].0
                } else {
                    elapsed_times[0].0
                };
                t += delta_t;
                n += delta_n;
                elapsed_times_reversed.push((n, t));
            }
            elapsed_times = elapsed_times_reversed;
        }
        BenchmarkMode::Find { recent } => {
            let mut total_elapsed = 0.0;
            for (i, x) in values.iter().enumerate() {
                insert(&mut set, *x);

                let len = i + 1;
                if len % params.measure_every == 0 {
                    let search_values_shuffled = if recent {
                        helpers::shuffle_clone(&values[len - params.measure_every .. len])
                    } else {
                        helpers::sample_clone(&values[.. len], params.measure_every)
                    };
                    assert_eq!(search_values_shuffled.len(), params.measure_every);

                    let start = Instant::now();
                    for x in search_values_shuffled {
                        assert!(find(&set, x));
                    }
                    total_elapsed += start.elapsed().as_secs_f64();
                    elapsed_times.push((len, total_elapsed));
                }
            }
            assert_eq!(get_len(&set), values.len());
        }
    }

    elapsed_times
}

type BenchFunc = Rc<dyn Fn(BenchmarkMode, BenchmarkParams, &[f64]) -> Vec<(usize, f64)>>;

struct AllBenches {
    bench_array_stump: BenchFunc,
    bench_splay_tree: BenchFunc,
    bench_b_tree: BenchFunc,
    bench_skiplist: BenchFunc,
    bench_slot_array: BenchFunc,
    bench_plain_array: BenchFunc,
}

impl AllBenches {
    fn new() -> Self {
        let bench_array_stump = |mode: BenchmarkMode, params: BenchmarkParams, values: &[f64]| {
            run_generic_benchmark(
                mode,
                params,
                &values,
                || ArrayStump::new_explicit(cmp_array_stump, 512),
                |set, x| set.insert(x).is_some(),
                |set, x| set.remove(&x),
                |set| set.len(),
                |set, x| set.find(&x).is_some(),
            )
        };
        let bench_splay_tree = |mode: BenchmarkMode, params: BenchmarkParams, values: &[f64]| {
            run_generic_benchmark(
                mode,
                params,
                &values,
                || SplaySet::new(cmp_splay_tree),
                |set, x| set.insert(x),
                |set, x| set.remove(&x),
                |set| set.len(),
                |set, x| set.find(&x).is_some(),
            )
        };
        let bench_b_tree = |mode: BenchmarkMode, params: BenchmarkParams, values: &[f64]| {
            run_generic_benchmark(
                mode,
                params,
                &values,
                BTreeSet::new,
                |set, x| set.insert(FloatWrapper(x)),
                |set, x| set.remove(&FloatWrapper(x)),
                |set| set.len(),
                |set, x| set.contains(&FloatWrapper(x)),
            )
        };
        let bench_skiplist = |mode: BenchmarkMode, params: BenchmarkParams, values: &[f64]| {
            run_generic_benchmark(
                mode,
                params,
                &values,
                || {
                    let mut skiplist = OrderedSkipList::<f64>::new();
                    unsafe {
                        skiplist.sort_by(cmp_skip_list);
                    }
                    skiplist
                },
                |set, x| {
                    set.insert(x);
                    true
                },
                |set, x| set.remove(&x).is_some(),
                |set| set.len(),
                |set, x| set.contains(&x),
            )
        };
        let bench_slot_array = |mode: BenchmarkMode, params: BenchmarkParams, values: &[f64]| {
            run_generic_benchmark(
                mode,
                params,
                &values,
                || SlotArray::new(cmp_slot_array, 20, 4),
                |set, x| set.insert(x),
                |set, x| set.remove(&x),
                |set| set.len(),
                |_set, _x| unimplemented!(),
            )
        };
        let bench_plain_array = |mode: BenchmarkMode, params: BenchmarkParams, values: &[f64]| {
            run_generic_benchmark(
                mode,
                params,
                &values,
                || PlainArray::new(cmp_plain_array, 1024),
                |set, x| set.insert(x),
                |set, x| set.remove(&x),
                |set| set.len(),
                |_set, _x| unimplemented!(),
            )
        };
        AllBenches {
            bench_array_stump: Rc::new(bench_array_stump),
            bench_splay_tree: Rc::new(bench_splay_tree),
            bench_b_tree: Rc::new(bench_b_tree),
            bench_skiplist: Rc::new(bench_skiplist),
            bench_slot_array: Rc::new(bench_slot_array),
            bench_plain_array: Rc::new(bench_plain_array),
        }
    }
}

#[derive(Clone)]
struct BenchmarkTask {
    name: String,
    func: BenchFunc,
    run: i32,
}

fn construct_benchmark_tasks(
    all_benches: &AllBenches,
    run: i32,
    all_combatants: bool,
) -> Vec<BenchmarkTask> {
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
        BenchmarkTask {
            run,
            name: "SkipList".to_string(),
            func: all_benches.bench_skiplist.clone(),
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


pub fn run_benchmarks(mode: BenchmarkMode, params: BenchmarkParams, gen_mode: GeneratorMode) {
    if cfg!(debug_assertions) {
        println!(
            "WARNING: Debug assertions are enabled. Benchmarking should be done in `--release`."
        );
    }
    println!("Running benchmark...");
    println!("    Benchmark mode: {}", mode);
    println!("    Generator mode: {}", gen_mode);
    println!("    N: {}", params.n);
    println!("    Measure every: {}", params.measure_every);
    println!("    Num runs: {}", params.num_runs);
    let n = params.n;

    let all_benches = AllBenches::new();

    for run in 0 ..= params.num_runs {
        let benchmark_tasks = construct_benchmark_tasks(&all_benches, run, params.all_combatants);

        let values = match gen_mode {
            GeneratorMode::Avg => helpers::gen_rand_values(n),
            GeneratorMode::Asc => (0 .. n).map(|x| x as f64 / n as f64).collect(),
            GeneratorMode::Dsc => (0 .. n).map(|x| x as f64 / n as f64).rev().collect(),
        };
        assert_eq!(values.len(), n);

        for benchmark_task in benchmark_tasks {
            println!(
                "Running benchmark task: {} / {}",
                benchmark_task.name, benchmark_task.run
            );

            let measurements = (benchmark_task.func)(mode, params, &values);

            // Use zero-th iteration for warm up
            if run > 0 {
                let iters: Vec<_> = measurements.iter().map(|i_t| i_t.0).collect();
                let times: Vec<_> = measurements.iter().map(|i_t| i_t.1).collect();
                helpers::export_elapsed_times(
                    &benchmark_task.name,
                    benchmark_task.run,
                    &mode.to_string(),
                    &gen_mode.to_string(),
                    params.n,
                    params.measure_every,
                    &format!(
                        "results/{}_{}_{}_{}.json",
                        mode, gen_mode, benchmark_task.name, benchmark_task.run
                    ),
                    &iters,
                    &times,
                );
            }
        }
    }

    println!("Num calls array stump: {:12}", get_num_calls_array_stump());
    println!("Num calls splay tree:  {:12}", get_num_calls_splay_tree());
    println!("Num calls B tree:      {:12}", get_num_calls_b_tree());
    println!("Num calls skip list    {:12}", get_num_calls_skip_list());
    println!("Num calls slot array:  {:12}", get_num_calls_slot_array());
    println!("Num calls plain array: {:12}", get_num_calls_plain_array());

    helpers::call_plots(&mode.to_string(), &gen_mode.to_string());
}
