# Oxide Chunk

**Oxide Chunk** provides GPU memory chunk management with ternary allocation status — `+1 (Allocated)`, `0 (Fragmented)`, `-1 (Free)` — featuring buddy-style splitting and merging, automatic coalescing on deallocation, defragmentation, and utilization statistics.

## Why It Matters

GPU memory is a scarce resource: a typical GPU has 8-24GB of VRAM shared by dozens of kernels. Without proper memory management, fragmentation — where free memory is split into many small, non-contiguous blocks — makes it impossible to allocate large buffers even when total free memory is sufficient. Oxide Chunk implements the buddy allocator algorithm (used by Linux's old zoned allocator and FreeBSD's uma zone allocator) with ternary status tracking. The buddy system guarantees that any allocation request that fits within total free memory will succeed after at most log₂(N) splits — bounded fragmentation in the worst case.

## How It Works

### Buddy Allocator

Memory starts as one large free chunk. On allocation, chunks split recursively:

```
allocate(size):
    find smallest free chunk ≥ size     — O(N) scan
    while chunk.size / 2 ≥ size:
        split chunk into two buddies    — O(1)
    mark left buddy as allocated
```

On deallocation, adjacent free buddies merge:

```
deallocate(offset):
    mark chunk as free
    coalesce():
        sort chunks by offset            — O(N log N)
        scan for adjacent free buddies   — O(N)
        merge until no more merges       — O(N) per pass
```

### Chunk Operations

```
Chunk { offset: u64, size: u64, alignment: u64, allocated: bool }

split() → Option<(Chunk, Chunk)>:
    half = size / 2
    left = Chunk { offset, size: half }
    right = Chunk { offset: offset + half, size: half }
    — only valid if size ≥ 2 and size % 2 == 0

can_merge(other) → bool:
    !self.allocated && !other.allocated
    && same alignment
    && adjacent (self.end == other.offset || other.end == self.offset)

merge(other) → Chunk:
    offset = min(self.offset, other.offset)
    size = self.size + other.size
```

Split/merge/can_merge: all **O(1)**.

### Ternary Status Classification

```
ternary_status() → TernaryStatus:
    has_allocated = any chunk.allocated
    has_free = any !chunk.allocated
    match (has_allocated, has_free):
        (true, false) → Allocated (+1)   // fully utilized
        (true, true)  → Fragmented (0)   // mixed — normal operating
        (false, _)    → Free (-1)        // empty pool
```

Status: **O(N)** single pass over chunks.

### Defragmentation

Compacts all allocated chunks to the beginning:

```
defragment():
    collect allocated chunks sorted by offset
    cursor = 0
    for each allocated chunk:
        move to cursor (respecting alignment)
        cursor += chunk.size
    create single free chunk from cursor to end
```

Defragmentation: **O(N log N)** (sort + rebuild). After defrag, fragmentation_ratio → 0.

### Fragmentation Metrics

```
utilization() = Σ allocated_sizes / total_size

fragmentation_ratio() = 1 - (largest_free_block / total_free)
    0.0 = no fragmentation (one big free block)
    1.0 = maximally fragmented (many tiny free blocks)

largest_free_block() = max(free chunk sizes)
```

All: **O(N)** single pass.

### Complexity Summary

| Operation | Cost |
|-----------|------|
| `allocate(size)` | O(N) best-fit search + O(log S) splits |
| `deallocate(offset)` | O(N) find + O(N log N) coalesce |
| `coalesce()` | O(N²) worst case, O(N) typical |
| `defragment()` | O(N log N) |
| `utilization()` | O(N) |
| `fragmentation_ratio()` | O(N) |
| `ternary_status()` | O(N) |

Where N = number of chunk entries (typically 10-1000).

## Quick Start

```rust
use oxide_chunk::{ChunkAllocator, TernaryStatus};

let mut alloc = ChunkAllocator::new(4096, 1); // 4KB pool, 1-byte align

// Allocate
let a = alloc.allocate(512).unwrap();
let b = alloc.allocate(256).unwrap();
println!("Utilization: {:.1}%", alloc.utilization() * 100);
println!("Status: {:?}", alloc.ternary_status()); // Fragmented

// Deallocate and coalesce
alloc.deallocate(a);
alloc.deallocate(b);
assert_eq!(alloc.ternary_status(), TernaryStatus::Free); // All merged back

// Defragment
let c = alloc.allocate(128).unwrap();
let d = alloc.allocate(128).unwrap();
alloc.deallocate(c);
alloc.defragment(); // Compact allocated chunks
println!("Fragmentation: {:.2}", alloc.fragmentation_ratio()); // ~0.0
```

## API

| Type | Key Methods | Description |
|------|-------------|-------------|
| `ChunkAllocator` | `new(total_size, alignment)`, `allocate(size) → Option<u64>`, `deallocate(offset) → bool` | Buddy-style allocator |
| `ChunkAllocator` | `coalesce()`, `defragment()`, `utilization()`, `fragmentation_ratio()`, `largest_free_block()`, `chunk_count()`, `chunks() → &[Chunk]` | Management and statistics |
| `Chunk` | `new(offset, size, alignment)`, `split()`, `can_merge(other)`, `merge(other)`, `end()` | Memory region |
| `TernaryStatus` | `Allocated (+1)`, `Fragmented (0)`, `Free (-1)` | Pool status enum |

## Architecture Notes

Oxide Chunk provides GPU memory management for SuperInstance. In γ + η = C, Allocated (+1) chunks represent γ (growth — memory actively used for computation), Free (-1) chunks represent η (avoidance — memory available but unused, potential for waste), and Fragmented (0) is the mixed state where the pool is neither fully utilized nor empty. The defragmentation operation directly conserves C: by compacting allocated chunks, it reduces wasted address space while preserving total allocated memory. Integrates with `page-allocator` for OS-level page management and `oxide-ring` for allocation event logging.

See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md) for GPU memory management architecture.

## References

1. Knowlton, K. C. (1965). "A Fast Storage Allocator." *Communications of the ACM*, 8(10), 623–625. (Original buddy system)
2. Knuth, D. E. (1997). *The Art of Computer Programming, Vol. 1*, 3rd ed. Section 2.5: Dynamic Storage Allocation.
3. Wilson, P. R. et al. (1995). "Dynamic Storage Allocation: A Survey and Critical Review." *International Workshop on Memory Management*. Springer.

## License

MIT OR Apache-2.0
