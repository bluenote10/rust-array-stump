# rust-array-stump

A data structure mixing dynamic array and sorted set semantics.

- insert / remove: O(sqrt N)
- rank index / next / prev: O(1)

This data structure is similar to [hashed array trees](https://en.wikipedia.org/wiki/Hashed_array_tree), but optimized for insert/remove in the middle instead of minimizing wasted space.

TODO...

## Benchmarks

TODO...

Other implementations:

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


