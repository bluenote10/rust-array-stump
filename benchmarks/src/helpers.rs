use std::cmp::Ordering;
use std::fs::{create_dir_all, File};
use std::path::Path;
use serde_json::json;

use std::process::{Command, Stdio};

use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};


#[macro_export]
macro_rules! create_cmp {
    ($func:ident, $get:ident, $count:ident) => {
        static mut $count: u64 = 0;

        // Inlining this function helps B tree performance a lot. Since
        // our use case is to simulate a situation of much more complex
        // compare functions (that are likely too big to be inlined),
        // let's enforce the non-inlined case.
        #[inline(never)]
        fn $func(a: &f64, b: &f64) -> std::cmp::Ordering {
            unsafe {
                $count += 1;
            }
            // Note the unnecessary exp calls are used to simulate a costly compare function
            a.exp().partial_cmp(&b.exp()).unwrap()
        }

        #[allow(dead_code)]
        pub fn $get() -> u64 {
            unsafe {
                $count
            }
        }
    };
}


#[derive(Debug)]
pub struct FloatWrapper(pub f64);

impl Eq for FloatWrapper {}

impl PartialEq for FloatWrapper {
    fn eq(&self, _other: &Self) -> bool {
        unimplemented!()
    }
}

impl PartialOrd for FloatWrapper {
    fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
        unimplemented!()
    }
}

create_cmp!(cmp_b_tree, get_num_calls_b_tree, NUM_CALLS_B_TREE);

impl Ord for FloatWrapper {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_b_tree(&self.0, &other.0)
    }
}


pub fn export_elapsed_times(name: &str, run: i32, filename: &str, iters: &[usize], times: &[f64]) {

    let json_data = json!({
        "name": name,
        "run": run,
        "iters": iters,
        "times": times,
    });

    let path = Path::new(filename);
    let parent = path.parent().unwrap();
    create_dir_all(parent).unwrap();

    let f = File::create(path).expect("Unable to create json file.");
    serde_json::to_writer_pretty(f, &json_data).expect("Unable to write json file.");
}


pub fn call_plots(prefix: &str) {
    let script_path = Path::new(file!()).to_path_buf()
        .canonicalize().unwrap()
        .parent().unwrap().to_path_buf() // -> /src
        .parent().unwrap().to_path_buf() // -> /
        .join("scripts")
        .join("plot.py");
    Command::new(script_path.as_os_str())
        .arg("-p")
        .arg(prefix)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .spawn()
        .expect("Failed to run Python plot.")
        .wait()
        .expect("Failed to wait for child");
}


pub fn export_stats(
    iters: &[usize],
    times: &[f64],
    fill_ratio: &[f64],
    fill_min: &[Option<usize>],
    fill_max: &[Option<usize>],
    num_blocks: &[usize],
    capacity: &[u16],
    num_cmp_calls: &[u64],
) {

    let json_data = json!({
        "iters": iters,
        "times": times,
        "fill_ratio": fill_ratio,
        "fill_min": fill_min,
        "fill_max": fill_max,
        "num_blocks": num_blocks,
        "capacity": capacity,
        "num_cmp_calls": num_cmp_calls,
    });

    let path = Path::new("results/fill_stats.json");
    let parent = path.parent().unwrap();
    create_dir_all(parent).unwrap();

    let f = File::create(path).expect("Unable to create json file.");
    serde_json::to_writer_pretty(f, &json_data).expect("Unable to write json file.");
}


pub fn call_plots_stats() {
    let script_path = Path::new(file!()).to_path_buf()
        .canonicalize().unwrap()
        .parent().unwrap().to_path_buf() // -> /src
        .parent().unwrap().to_path_buf() // -> /
        .join("scripts")
        .join("plot_stats.py");
    Command::new(script_path.as_os_str())
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .spawn()
        .expect("Failed to run Python plot.")
        .wait()
        .expect("Failed to wait for child");
}


pub fn gen_rand_values(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let values: Vec<f64> = (0..n).map(|_| rng.gen()).collect();
    values
}


pub fn gen_rand_values_i32(n: usize) -> Vec<i32> {
    let mut rng = rand::thread_rng();
    let values: Vec<i32> = (0..n).map(|_| rng.gen_range(0, 32)).collect();
    values
}


pub fn shuffle<T>(v: &mut Vec<T>) -> &mut Vec<T>
{
    let mut rng = thread_rng();
    v.shuffle(&mut rng);
    v
}