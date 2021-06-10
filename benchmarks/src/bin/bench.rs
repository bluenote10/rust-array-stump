use clap::{App, AppSettings, Arg};

use array_stump_benchmarks::benchmarks::{
    run_benchmarks, BenchmarkMode, BenchmarkParams, GeneratorMode,
};

fn main() {
    #[rustfmt::skip]
    let matches = App::new("Benchmark runner")
        .setting(AppSettings::ArgRequiredElseHelp)
        .arg(Arg::with_name("bench-mode")
                 .long("bench-mode")
                 .short("b")
                 .default_value("insert")
                 .possible_values(&["insert", "remove", "find_rand", "find_recent"])
                 .help("Benchmark mode"))
        .arg(Arg::with_name("gen-mode")
                 .long("gen-mode")
                 .short("g")
                 .default_value("avg")
                 .possible_values(&["avg", "asc", "dsc"])
                 .help("Generator mode"))
        .get_matches();

    let bench_mode = matches.value_of("bench-mode").unwrap().to_string();
    let gen_mode = matches.value_of("gen-mode").unwrap().to_string();

    let bench_mode = match bench_mode.as_ref() {
        "insert" => BenchmarkMode::Insert,
        "remove" => BenchmarkMode::Remove,
        "find_rand" => BenchmarkMode::Find { recent: false },
        "find_recent" => BenchmarkMode::Find { recent: true },
        _ => panic!("Illegal benchmark mode"),
    };
    let gen_mode = match gen_mode.as_ref() {
        "avg" => GeneratorMode::Avg,
        "asc" => GeneratorMode::Asc,
        "dsc" => GeneratorMode::Dsc,
        _ => panic!("Illegal benchmark mode"),
    };

    let bench_params = BenchmarkParams {
        n: 1000000,
        measure_every: 25,
        num_runs: 3,
        all_combatants: false,
    };

    run_benchmarks(bench_mode, bench_params, gen_mode);
}
