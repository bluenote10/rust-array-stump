#!/bin/bash

# to avoid crashes in the Splay tree...
ulimit -s 32768

cargo run --release --bin bench -- -b insert -g avg
cargo run --release --bin bench -- -b insert -g asc
cargo run --release --bin bench -- -b insert -g dsc

cargo run --release --bin bench -- -b remove

cargo run --release --bin bench -- -b find_rand
cargo run --release --bin bench -- -b find_recent
