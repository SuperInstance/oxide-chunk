# oxide-chunk

*GPU memory chunk management with ternary allocation status. +1 = allocated, 0 = fragmented, -1 = free. Buddy-style splitting, coalescing, and defragmentation — because GPU memory is too expensive to waste on fragmentation.*

## Why This Exists

GPU memory is scarce. An RTX 4050 has 6 GB — that's it. Every fragment of wasted memory is a kernel that can't launch, a batch that has to shrink, or a model that can't fit. Traditional binary allocators (used/free) can't distinguish between "free and usable" and "technically free but riddled with internal holes."

The ternary status adds a third state: **fragmented** (0). A fragmented chunk has free space, but it's not contiguous enough to satisfy a new allocation. The defragmenter coalesces these back into usable blocks.

## Architecture

```
6 GB GPU Memory
├── Chunk 0: +1 (allocated, 256 MB)
├── Chunk 1: 0 (fragmented, 256 MB, 40% free but split)
├── Chunk 2: -1 (free, 512 MB)
├── Chunk 3: +1 (allocated, 1 GB)
└── ...

Buddy Split:  512 MB → 256 + 256
Buddy Merge:  256 + 256 → 512 (if both free)
Defragment:   Fragmented 256 MB → Coalesce free regions → Usable block
```

### Key Types

- **`Chunk`** — Memory block with base address, size, and ternary status.
- **`ChunkAllocator`** — Buddy-style allocator. Splits blocks on allocation, merges on free. Tracks fragmentation per chunk.
- **`Defragmenter`** — Coalesces fragmented chunks. Returns recovered bytes and new free blocks.
- **`UtilizationStats`** — Total / used / fragmented / free bytes. Fragmentation ratio. Largest contiguous free block.

## Usage

```rust
use oxide_chunk::*;

let mut alloc = ChunkAllocator::new(6 * 1024 * 1024 * 1024); // 6 GB

// Allocate for a kernel
let block = alloc.allocate(256 * 1024 * 1024).unwrap();
assert_eq!(block.status(), ChunkStatus::Allocated);

// Free it
alloc.deallocate(block);
// Adjacent free blocks auto-merge via buddy system

// Check fragmentation
let stats = alloc.utilization();
println!("Fragmented: {:.1}%", stats.fragmentation_ratio() * 100);
println!("Largest free: {} MB", stats.largest_free() / (1024*1024));

// Defragment if needed
if stats.fragmentation_ratio() > 0.3 {
    let recovered = alloc.defragment();
    println!("Recovered {} MB", recovered / (1024*1024));
}
```

## The Deeper Idea

The ternary state (allocated/fragmented/free) is a microcosm of the SuperInstance resource management philosophy: binary categorization loses information that matters. In `agent-ensemble`, the same pattern appears — agents aren't just "active" or "idle," they're "active," "warming up," or "cooling down." The third state captures the transitional reality that binary abstractions erase.

For GPU memory specifically, the fragmented state enables *proactive* defragmentation. Instead of waiting for an OOM error and then desperately compacting, the allocator can defragment during idle periods — between kernel launches, during pipeline stalls. This is the same principle as `agent-rubato`'s tempo flexibility: use the pauses productively.

## Related Crates

- `oxide-epoch` — Epoch-based reclamation for safely freeing chunk memory
- `oxide-ring` — Ring buffers that allocate from chunks
- `oxide-sandbox` — Sandboxed execution with chunk-based memory isolation
- `oxide-tenancy` — Multi-tenant GPU allocation built on chunks
