
use clap::{App, AppSettings, Arg};

use array_stump_benchmarks::create_cmp;
use array_stump_benchmarks::helpers;

use array_stump::ArrayStump;

use std::time::Instant;
use pretty_assertions::assert_eq;

create_cmp!(cmp_array_tree, get_num_calls_array_tree, NUM_CALLS_ARRAY_TREE);

fn run_fill_statistics() {
    #[rustfmt::skip]
    let matches = App::new("Benchmark runner")
        .setting(AppSettings::ArgRequiredElseHelp)
        .arg(Arg::with_name("gen-mode")
                 .long("gen-mode")
                 .short("g")
                 .default_value("avg")
                 .possible_values(&["avg", "asc", "dsc"])
                 .help("Generator mode"))
        .get_matches();

    let gen_mode = matches.value_of("gen-mode").unwrap().to_string();

    let n = 1000000;
    let measure_every = 10;
    let values = match gen_mode.as_ref() {
        "avg" => helpers::gen_rand_values(n),
        "asc" => (0 .. n).map(|x| x as f64 / n as f64).collect(),
        "dsc" => (0 .. n).map(|x| x as f64 / n as f64).rev().collect(),
        _ => panic!("Invalid generator mode")
    };

    let mut set = ArrayStump::new(cmp_array_tree, 256);

    let mut iters = Vec::new();
    let mut times = Vec::new();
    let mut fill_ratio = Vec::new();
    let mut fill_min = Vec::new();
    let mut fill_max = Vec::new();
    let mut num_blocks = Vec::new();
    let mut capacity = Vec::new();
    let mut num_cmp_calls = Vec::new();

    println!("Inserting...");
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

    println!("Removing...");
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

    println!("Exporting...");
    helpers::export_stats(&iters, &times, &fill_ratio, &fill_min, &fill_max, &num_blocks, &capacity, &num_cmp_calls);
    helpers::call_plots_stats();
}

fn main() {
    run_fill_statistics();
}
