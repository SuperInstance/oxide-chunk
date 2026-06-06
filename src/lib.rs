//! # oxide-chunk
//!
//! GPU memory chunk management with ternary allocation status.
//!
//! Ternary status: `+1` = allocated, `0` = fragmented, `-1` = free.
//! Features buddy-style splitting/merging, coalescing, defragmentation, and utilization stats.

use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Ternary status
// ---------------------------------------------------------------------------

/// Ternary allocation status for a memory chunk or region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TernaryStatus {
    /// All allocated – large contiguous block (`+1`).
    Allocated,
    /// Mix of allocated and free – fragmented (`0`).
    Fragmented,
    /// Entirely free (`-1`).
    Free,
}

impl From<i8> for TernaryStatus {
    fn from(v: i8) -> Self {
        match v.cmp(&0) {
            Ordering::Greater => TernaryStatus::Allocated,
            Ordering::Equal => TernaryStatus::Fragmented,
            Ordering::Less => TernaryStatus::Free,
        }
    }
}

impl From<TernaryStatus> for i8 {
    fn from(s: TernaryStatus) -> Self {
        match s {
            TernaryStatus::Allocated => 1,
            TernaryStatus::Fragmented => 0,
            TernaryStatus::Free => -1,
        }
    }
}

// ---------------------------------------------------------------------------
// Chunk
// ---------------------------------------------------------------------------

/// A contiguous region of GPU memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Byte offset from the start of the memory pool.
    pub offset: u64,
    /// Size in bytes.
    pub size: u64,
    /// Alignment requirement in bytes (must be a power of two).
    pub alignment: u64,
    /// Whether this chunk is currently allocated.
    pub allocated: bool,
}

impl Chunk {
    /// Create a new free chunk.
    pub fn new(offset: u64, size: u64, alignment: u64) -> Self {
        assert!(alignment.is_power_of_two() || alignment == 0);
        Self {
            offset,
            size,
            alignment: alignment.max(1),
            allocated: false,
        }
    }

    /// End address (exclusive).
    pub fn end(&self) -> u64 {
        self.offset + self.size
    }

    /// Split this chunk into two buddy halves. Returns `None` if size is not
    /// evenly divisible or would produce zero-sized halves.
    pub fn split(&self) -> Option<(Chunk, Chunk)> {
        if self.size < 2 {
            return None;
        }
        let half = self.size / 2;
        if half == 0 {
            return None;
        }
        // Only split if size is a power of two or evenly divisible.
        if self.size % 2 != 0 {
            return None;
        }
        let left = Chunk {
            offset: self.offset,
            size: half,
            alignment: self.alignment,
            allocated: false,
        };
        let aligned_offset = (self.offset + half + self.alignment - 1) & !(self.alignment - 1);
        let right_offset = aligned_offset.max(self.offset + half);
        let right_size = self.end().saturating_sub(right_offset);
        if right_size == 0 {
            return None;
        }
        let right = Chunk {
            offset: right_offset,
            size: right_size,
            alignment: self.alignment,
            allocated: false,
        };
        Some((left, right))
    }

    /// Can this chunk be buddy-merged with `other`?
    pub fn can_merge(&self, other: &Chunk) -> bool {
        !self.allocated
            && !other.allocated
            && self.alignment == other.alignment
            && (self.end() == other.offset || other.end() == self.offset)
    }

    /// Merge with `other`, producing a single free chunk.
    pub fn merge(&self, other: &Chunk) -> Chunk {
        let offset = self.offset.min(other.offset);
        let size = self.size + other.size;
        Chunk {
            offset,
            size,
            alignment: self.alignment,
            allocated: false,
        }
    }
}

// ---------------------------------------------------------------------------
// ChunkAllocator
// ---------------------------------------------------------------------------

/// Buddy-style GPU memory chunk allocator.
#[derive(Debug, Clone)]
pub struct ChunkAllocator {
    /// Total memory pool size in bytes.
    total_size: u64,
    /// Default alignment for new chunks.
    default_alignment: u64,
    /// Managed chunks, kept sorted by offset.
    chunks: Vec<Chunk>,
}

impl ChunkAllocator {
    /// Create a new allocator managing `total_size` bytes with the given
    /// default alignment.
    pub fn new(total_size: u64, default_alignment: u64) -> Self {
        let alignment = default_alignment.max(1);
        let initial = Chunk::new(0, total_size, alignment);
        Self {
            total_size,
            default_alignment: alignment,
            chunks: vec![initial],
        }
    }

    /// Allocate `size` bytes using buddy-style splitting (first-fit).
    /// Returns the offset of the allocated block, or `None` if no suitable
    /// free chunk exists.
    pub fn allocate(&mut self, size: u64) -> Option<u64> {
        // Find the smallest free chunk that fits.
        let mut best_idx = None;
        let mut best_size = u64::MAX;
        for (i, c) in self.chunks.iter().enumerate() {
            if !c.allocated && c.size >= size && c.size < best_size {
                best_idx = Some(i);
                best_size = c.size;
                if best_size == size {
                    break;
                }
            }
        }
        let idx = best_idx?;

        // Buddy-split until close to requested size or can't split further.
        loop {
            let chunk = &self.chunks[idx];
            if chunk.size / 2 < size || chunk.size % 2 != 0 {
                break;
            }
            let (left, right) = match chunk.split() {
                Some(pair) => pair,
                None => break,
            };
            self.chunks.remove(idx);
            self.chunks.insert(idx, left);
            self.chunks.insert(idx + 1, right);
            // Both are free; keep looking at left half.
        }

        let chunk = &mut self.chunks[idx];
        if chunk.size < size {
            return None;
        }
        chunk.allocated = true;
        Some(chunk.offset)
    }

    /// Free the chunk at `offset`.
    pub fn deallocate(&mut self, offset: u64) -> bool {
        if let Some(c) = self.chunks.iter_mut().find(|c| c.offset == offset && c.allocated) {
            c.allocated = false;
            self.coalesce();
            true
        } else {
            false
        }
    }

    /// Coalesce adjacent free chunks (buddy merging).
    pub fn coalesce(&mut self) {
        self.chunks.sort_by_key(|c| c.offset);
        let mut merged = true;
        while merged {
            merged = false;
            let mut new_chunks: Vec<Chunk> = Vec::with_capacity(self.chunks.len());
            let mut i = 0;
            while i < self.chunks.len() {
                if i + 1 < self.chunks.len() && self.chunks[i].can_merge(&self.chunks[i + 1]) {
                    new_chunks.push(self.chunks[i].merge(&self.chunks[i + 1]));
                    merged = true;
                    i += 2;
                } else {
                    new_chunks.push(self.chunks[i].clone());
                    i += 1;
                }
            }
            self.chunks = new_chunks;
        }
    }

    /// Defragment: compact all allocated chunks to the beginning of the pool,
    /// leaving a single free chunk at the end.
    pub fn defragment(&mut self) {
        let allocated: Vec<&Chunk> = self.chunks.iter().filter(|c| c.allocated).collect();
        if allocated.is_empty() {
            // Everything is free – collapse into a single chunk.
            self.chunks = vec![Chunk::new(0, self.total_size, self.default_alignment)];
            return;
        }
        let mut new_chunks: Vec<Chunk> = Vec::new();
        let mut cursor: u64 = 0;
        for chunk in &allocated {
            let aligned = (cursor + chunk.alignment - 1) & !(chunk.alignment - 1);
            new_chunks.push(Chunk {
                offset: aligned,
                size: chunk.size,
                alignment: chunk.alignment,
                allocated: true,
            });
            cursor = aligned + chunk.size;
        }
        if cursor < self.total_size {
            new_chunks.push(Chunk::new(cursor, self.total_size - cursor, self.default_alignment));
        }
        self.chunks = new_chunks;
    }

    /// Classify the overall allocation status of the pool.
    pub fn ternary_status(&self) -> TernaryStatus {
        let allocated_any = self.chunks.iter().any(|c| c.allocated);
        let free_any = self.chunks.iter().any(|c| !c.allocated);
        match (allocated_any, free_any) {
            (true, false) => TernaryStatus::Allocated,
            (true, true) => TernaryStatus::Fragmented,
            (false, _) => TernaryStatus::Free,
        }
    }

    /// Current utilization: ratio of allocated bytes to total bytes.
    pub fn utilization(&self) -> f64 {
        if self.total_size == 0 {
            return 0.0;
        }
        let allocated: u64 = self.chunks.iter().filter(|c| c.allocated).map(|c| c.size).sum();
        allocated as f64 / self.total_size as f64
    }

    /// Fragmentation ratio: 1.0 means maximally fragmented.
    /// Defined as `1 - (largest_free_block / total_free)`.
    pub fn fragmentation_ratio(&self) -> f64 {
        let free_chunks: Vec<&Chunk> = self.chunks.iter().filter(|c| !c.allocated).collect();
        if free_chunks.is_empty() {
            return 0.0;
        }
        if free_chunks.len() == 1 {
            return 0.0;
        }
        let total_free: u64 = free_chunks.iter().map(|c| c.size).sum();
        let largest: u64 = free_chunks.iter().map(|c| c.size).max().unwrap_or(0);
        if total_free == 0 {
            return 0.0;
        }
        1.0 - (largest as f64 / total_free as f64)
    }

    /// Size of the largest free block in bytes.
    pub fn largest_free_block(&self) -> u64 {
        self.chunks
            .iter()
            .filter(|c| !c.allocated)
            .map(|c| c.size)
            .max()
            .unwrap_or(0)
    }

    /// Number of chunks currently tracked.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Reference to the internal chunks slice (for inspection / tests).
    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_status_from_i8() {
        assert_eq!(TernaryStatus::from(1i8), TernaryStatus::Allocated);
        assert_eq!(TernaryStatus::from(5i8), TernaryStatus::Allocated);
        assert_eq!(TernaryStatus::from(0i8), TernaryStatus::Fragmented);
        assert_eq!(TernaryStatus::from(-1i8), TernaryStatus::Free);
        assert_eq!(TernaryStatus::from(-3i8), TernaryStatus::Free);
    }

    #[test]
    fn test_ternary_status_to_i8() {
        assert_eq!(i8::from(TernaryStatus::Allocated), 1);
        assert_eq!(i8::from(TernaryStatus::Fragmented), 0);
        assert_eq!(i8::from(TernaryStatus::Free), -1);
    }

    #[test]
    fn test_allocate_and_deallocate() {
        let mut alloc = ChunkAllocator::new(1024, 1);
        let offset = alloc.allocate(256).unwrap();
        assert_eq!(offset, 0);
        assert_eq!(alloc.ternary_status(), TernaryStatus::Fragmented);
        assert!(alloc.deallocate(offset));
        assert_eq!(alloc.ternary_status(), TernaryStatus::Free);
        assert_eq!(alloc.chunk_count(), 1);
    }

    #[test]
    fn test_buddy_splitting() {
        let mut alloc = ChunkAllocator::new(1024, 1);
        // 1024 → split to get something close to 256
        let a = alloc.allocate(256).unwrap();
        let b = alloc.allocate(256).unwrap();
        // Both should be at distinct offsets
        assert_ne!(a, b);
        // After allocating 2×256 from a 1024 pool, utilization should be 0.5
        assert!((alloc.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_coalescing() {
        let mut alloc = ChunkAllocator::new(512, 1);
        let a = alloc.allocate(256).unwrap();
        let b = alloc.allocate(256).unwrap();
        assert_eq!(alloc.chunk_count(), 2);
        alloc.deallocate(a);
        alloc.deallocate(b);
        // After coalescing, should be back to a single free chunk
        assert_eq!(alloc.chunk_count(), 1);
        assert_eq!(alloc.chunks()[0].size, 512);
    }

    #[test]
    fn test_defragmentation() {
        let mut alloc = ChunkAllocator::new(1024, 1);
        let a = alloc.allocate(256).unwrap();
        let _b = alloc.allocate(256).unwrap();
        alloc.deallocate(a);
        // Now we have a free chunk, an allocated chunk, and another free chunk
        let free_before = alloc.chunks().iter().filter(|c| !c.allocated).count();
        assert!(free_before >= 1);

        alloc.defragment();
        // After defrag: one allocated chunk at offset 0, one free at the end
        assert_eq!(alloc.chunk_count(), 2);
        assert!(alloc.chunks()[0].allocated);
        assert!(!alloc.chunks()[1].allocated);
    }

    #[test]
    fn test_fragmentation_ratio() {
        let mut alloc = ChunkAllocator::new(1024, 1);
        // Allocate every other 128-byte block to create fragmentation
        let a = alloc.allocate(256).unwrap();
        let b = alloc.allocate(256).unwrap();
        alloc.deallocate(a);
        // Free chunk + allocated chunk → some fragmentation
        let ratio = alloc.fragmentation_ratio();
        assert!(ratio >= 0.0 && ratio <= 1.0);
    }

    #[test]
    fn test_largest_free_block() {
        let mut alloc = ChunkAllocator::new(1024, 1);
        alloc.allocate(256);
        let largest = alloc.largest_free_block();
        // Should have some free space left
        assert!(largest > 0);
        assert!(largest <= 1024);
    }

    #[test]
    fn test_full_allocation() {
        let mut alloc = ChunkAllocator::new(512, 1);
        let a = alloc.allocate(512).unwrap();
        assert_eq!(alloc.ternary_status(), TernaryStatus::Allocated);
        assert!((alloc.utilization() - 1.0).abs() < f64::EPSILON);
        assert_eq!(alloc.largest_free_block(), 0);
        alloc.deallocate(a);
        assert_eq!(alloc.ternary_status(), TernaryStatus::Free);
    }

    #[test]
    fn test_out_of_memory() {
        let mut alloc = ChunkAllocator::new(128, 1);
        assert!(alloc.allocate(128).is_some());
        assert!(alloc.allocate(1).is_none());
    }
}
