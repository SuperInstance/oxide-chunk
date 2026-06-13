# Oxide Chunk

**Oxide Chunk** provides GPU memory chunk management with ternary allocation status — `+1` (allocated), `0` (fragmented), `-1` (free) — implementing buddy-system splitting, merging, coalescing, and defragmentation for efficient GPU memory utilization.

## Why It Matters

GPU memory is scarce and expensive — an H100 has 80GB of HBM3 at $30k+. Fragmentation can waste 20-30% of available memory, causing out-of-memory errors even when sufficient free memory exists in fragmented chunks. The buddy allocator solves this by maintaining power-of-two sized blocks that split and merge cleanly. Oxide Chunk adds ternary status tracking: not just free/allocated, but fragmented — indicating a block has some free and some allocated sub-blocks, enabling smarter allocation decisions.

## How It Works

### Buddy Allocator

Memory is divided into blocks of size `2^order` pages. The free list has one entry per order:

```
Order 0: [4KB]  [4KB] [4KB] ...
Order 1: [8KB]        [8KB] ...
Order 2:       [16KB] ...
...
Order 10: [4MB]
```

**Allocation** (request size S):
1. Compute order = ceil(log2(S / page_size))
2. Find smallest free block at that order or above
3. Split larger blocks down, updating free lists
4. Return pointer, mark as Allocated (+1)

Cost: **O(log N)** where N = max order.

**Deallocation**:
1. Mark block as Free (-1)
2. Check if buddy (address XOR block_size) is also Free
3. If so, merge into parent block (repeat recursively)
4. Update free lists

Cost: **O(log N)** for coalescing.

### Ternary Status

Each chunk tracks one of three states:
- **Allocated (+1)**: Entirely in use
- **Fragmented (0)**: Partially used — has sub-blocks in different states
- **Free (-1)**: Entirely available

Fragmentation detection: **O(1)** per block (compare allocated_bytes vs capacity).

### Defragmentation

The defragmentation pass scans all chunks and compacts allocations:

```
for each fragmented chunk:
    move allocated sub-blocks to contiguous free space
    merge freed regions
    update status to Free or Allocated
```

Cost: **O(N)** where N = number of allocated blocks. Triggers when fragmentation ratio exceeds threshold.

### Utilization Statistics

```
utilization = allocated_bytes / total_bytes
fragmentation = 1 - (largest_free_block / total_free)
```

Both: **O(1)** to compute from tracked counters.

## Quick Start

```rust
use oxide_chunk::{ChunkManager, TernaryStatus};

let mut mgr = ChunkManager::new(1024 * 1024); // 1MB pool
let ptr1 = mgr.alloc(4096).unwrap();
let ptr2 = mgr.alloc(8192).unwrap();

println!("Status: {:?}", mgr.chunk_status(ptr1)); // Allocated

mgr.dealloc(ptr1);
mgr.dealloc(ptr2);
mgr.defragment(); // Coalesce and compact
```

## API

| Type | Description |
|------|-------------|
| `TernaryStatus` | `Allocated (+1)`, `Fragmented (0)`, `Free (-1)` |
| `ChunkManager` | Buddy allocator with ternary tracking |
| `Chunk` | Individual memory chunk with status and size |

Key methods: `alloc(size)`, `dealloc(ptr)`, `defragment()`, `utilization()`, `fragmentation_ratio()`.

## Architecture Notes

Oxide Chunk is part of the oxide-* GPU memory management stack. In γ + η = C, allocation is γ (growth — expanding to meet compute demands) while deallocation and defragmentation are η (avoidance — reclaiming wasted space and preventing fragmentation-induced OOM). Works with `oxide-epoch` for safe reclamation and `oxide-ring` for buffer management.

See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md) for GPU memory architecture.

## References

1. Knuth, D. E. (1973). *The Art of Computer Programming, Vol. 1*, 3rd ed. Section 2.5: Dynamic Storage Allocation. Addison-Wesley.
2. Knowlton, K. C. (1965). "A Fast Storage Allocator." *Communications of the ACM*, 8(10), 623–625.
3. NVIDIA (2024). "CUDA Memory Management Best Practices." *NVIDIA Developer Documentation*.

## License

MIT
