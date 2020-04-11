# rust-array-stump

--------------------------

[![Build Status](https://travis-ci.org/bluenote10/rust-array-stump.svg?branch=master)](https://travis-ci.org/bluenote10/rust-array-stump)

A data structure mixing dynamic array and sorted set semantics.

- insert / remove: O(sqrt N)
- rank index / next / prev: O(1)

This data structure is similar to [hashed array trees](https://en.wikipedia.org/wiki/Hashed_array_tree), but optimized for insert/remove in the middle instead of minimizing wasted space.

TODO...

## Benchmarks

Benchmarks... as always take with a grain of salt. A few general notes:

- The benchmarks use `T = f64` as underlying set type.

- In order to simulate the use case in rust-geo-booleanop of having a very complex comparison function, the float comparison has been artificially slowed down by (1) preventing inlining, and (2) two unnecessary `f64.exp()` calls. This imposes a penalty in particular for `std::collections::BTreeSet` which states in the docs:

    > Currently, our implementation simply performs naive linear search. This provides excellent performance on small nodes of elements which are cheap to compare.

    This can be reproduced in the benchmarks: When switching to a cheap comparison function, BTreeSet is by far the fastest implementation.

- In general the benchmarks measure the time of micro batches. For instance, in the 'insert' benchmark, the data structures get filled with N = 1 million elements, and the elapsed time is measured every k = 25 elements.

- Several of these runs are performed for statistical stability. This shows as a "bundle" of lines in the plots. As can be seen, results between runs are quite consistent and typically 3 runs were sufficient.

Comparison data structures:

- [std::collections::BTreeSet](https://doc.rust-lang.org/std/collections/struct.BTreeSet.html)
- [SplayTree](https://github.com/21re/rust-geo-booleanop/tree/master/lib/src/splay): The implementation currently used in [rust-geo-booleanop](https://github.com/21re/rust-geo-booleanop), adapted from [alexcrichton/splay-rs](https://github.com/alexcrichton/splay-rs)
- [SkipList](https://docs.rs/skiplist/0.3.0/skiplist/)


### Insert (random)

![image](results/insert_avg_comparison.png/)

### Insert (ascending)

![image](results/insert_asc_comparison.png/)

### Insert (descending)

![image](results/insert_dsc_comparison.png/)

### Remove (random)

![image](results/remove_avg_comparison.png/)

### Find (random)

![image](results/find_rand_avg_comparison.png/)

### Find (recent)

![image](results/find_recent_avg_comparison.png/)


