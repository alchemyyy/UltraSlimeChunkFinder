# Ultra Slime Chunk Finder

GPU-accelerated Minecraft slime chunk pattern finder.

<img width="350" height="350" alt="image" src="https://github.com/user-attachments/assets/1f8b3ef6-5d4d-4060-9897-77f9d111217f" />

https://www.chunkbase.com/apps/slime-finder#seed=100103745034619&platform=java&x=54483&z=96957&zoom=1.281

In terms of "raw probability", the odds of an N by N brick of slime chunks is 1/(10^(N^2)). For N=4 (example above), this is 1 in 10 quadrillion.

slime.cu searches ~16000 positions per second (across all seeds at each position) on an RTX 4080 Super.

I think big slime chunks are very satisfying, so I built a set of utilities to search for them.

TODO: 
 * Z has consistently been exhibiting periodicity
 * GPU batching is now required (k-stride made things a lot more efficient than they used to be)
 * Finish adding "scaling" to slime.cu. right now it is only confirmed working for 4x4's.
 * Investigate universal pattern possibility with hensel + k stride algo.


## Build

Requires CUDA Toolkit and Visual Studio.

```
build.bat          # builds slime.exe
build_fft.bat      # builds slimefft.exe
build_world.bat    # builds slime_world.exe
buildrefiner.bat   # builds slime_refiner.exe
etc..
```

## FFT Pattern Format

Used by slime_fft and slime_world:

```
#  = must be slime chunk
.  = must NOT be slime chunk
X  = don't care (wildcard)
```

Example - isolated 3x3:
```
.....
.###.
.###.
.###.
.....
```

---

## slime

Finds all seeds containing a solid NxN slime pattern at any position.

```
slime.exe [--radius N] [--start X Z]
```

Pattern size set at compile time (`PATTERN_SIZE` constant, default 4×4).

---

### Math

#### The Slime Chunk Formula

Minecraft determines if chunk (x, z) is a slime chunk like so:

```
slimeSeed = seed + x²·A + x·B + z²·C + z·D
internal  = (slimeSeed ⊕ COMBINED_XOR) & 0xFFFFFFFFFFFF   (48-bit mask)
advanced  = (internal · MULT + ADD) & 0xFFFFFFFFFFFF
isSlime   = (advanced >> 17) mod 10 == 0
```

Here are the constants:
```
MULT         = 0x5DEECE66D  (25214903917)
ADD          = 0xB          (11)
XOR_CONST    = 0x3AD8025F
COMBINED_XOR = XOR_CONST ⊕ MULT = 0x5E434E432

A = 0x4C1906   (4987142)
B = 0x5AC0DB   (5947611)  
C = 0x4307A7   (4392871)
D = 0x5F24F    (389711)
```

Note: `(advanced >> 17) mod 10` extracts bits 17-47 of the LCG
result. Lower seed bits propagate upward through multiplication, creating a
dependency structure we can exploit.

#### Seed Space Visualization

```
48-bit seed structure:
┌─────────────────────────────────────────────────────────────────┐
│ bit 47                                               bit 0      │
├────────────────────────────┬────────────────────────────────────┤
│      UPPER BITS (k)        │         ROOT (lower bits)          │
│      27 bits for 4×4       │         21 bits for 4×4            │
│   (enumerate with stride)  │      (solve via Hensel lifting)    │
└────────────────────────────┴────────────────────────────────────┘
                             │
                         ROOT_BITS boundary
```

For an N×N pattern:
- More constraints → more bits "locked" by the system
- `ROOT_BITS ≈ 17 + log₂(N²)` empirically
- We solve for roots, then enumerate upper bits

---

### Algorithm Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 1: HENSEL LIFTING                     │
│                                                                 │
│   Find all "roots" - lower ROOT_BITS that could produce the    │
│   pattern. Reduces 2^48 search to ~7,500 candidate roots.       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 2: K-OFFSET DISCOVERY                     │
│                                                                 │
│   For each root, find the k-offset: the value k₀ ∈ [0, 16384)  │
│   where seed = root + k₀·2^ROOT_BITS produces the pattern.      │
│   Most roots have NO valid k₀ (false positives from Hensel).    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               PHASE 3: STRIDED ENUMERATION                      │
│                                                                 │
│   For roots with valid k₀, enumerate k = k₀ + n·STRIDE          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Phase 1: Hensel Lifting

The slime check output depends on `(advanced >> 17) mod 10`. This means:
- Bits 0-16 of `advanced` don't affect the result
- Bit k of the seed first influences output at bit position ~(k + some offset)

We can determine valid lower bits incrementally: if bit k doesn't satisfy the
constraints, no extension to higher bits will fix it.

#### The Achievability Condition

For a partial seed with bits 0..(k-1) determined, we compute a "residue":

```
residue = (partial_advanced >> 17) mod 10
```

This residue must be "achievable" - able to reach 0 when higher bits are added.
When we add bit k (value 0 or 1), the residue changes by:

```
Δ = (2^k · MULT >> 17) mod 10
```

A residue r is achievable at bit level k if there exists m ∈ {0,1,...,9} such that:
```
(r + m · Δ) mod 10 = 0
```

The achievable residues depend on Δ:
```
Δ mod 10  │  Achievable residues
──────────┼─────────────────────────
    2     │  {0, 2, 4, 6, 8}  (evens)
    4     │  {0, 2, 4, 6, 8}  (evens)
    6     │  {0, 2, 4, 6, 8}  (evens)
    8     │  {0, 2, 4, 6, 8}  (evens)
    5     │  {0, 5}
    1,3,7,9│ {0,1,2,3,4,5,6,7,8,9} (all)
```

#### Hensel Lifting Procedure

```
                    HENSEL LIFTING VISUALIZATION
                    
Bit 18:   Start with all 2^18 = 262,144 candidates
          
          ████████████████████████████████████████████  262,144
          
          For each: check if all 16 chunk residues achievable
          
          ██████████████████████████████              ~140,000 survive

Bit 19:   Try appending 0 and 1 to each survivor (×2)
          
          ████████████████████████████████████████████  280,000 tried
          
          ██████████████████████████████████████       ~200,000 survive

Bit 20:   
          ████████████████████████████████████████████████████████  400,000 tried
          
          ██████████████████████████████████████████████████        ~280,000 survive

Bit 21:   (ROOT_BITS for 4×4)
          
          ████████████████████████████████████████████████████████████████  560,000 tried
          
          █████████                                                         ~7,500 roots
```

The filtering becomes more aggressive as we add bits because:
1. More bits → more precise residue calculation
2. All 16 constraints must remain satisfiable simultaneously
3. The constraints become increasingly correlated

#### Why ~7,500 Roots for 4×4?

Each slime constraint provides ~log₂(10) ≈ 3.32 bits of information.
With 16 constraints: 16 × 3.32 ≈ 53 bits of constraint.

But constraints share dependencies. The effective constraint is ~21 bits,
leaving 2^48 / 2^21 ≈ 2^27 possible seeds per position, distributed across
~7,500 roots.

In reality this can and does swing up and down from this value.

---

### Phase 2: K-Offset

For a given position and root, define `k` as the upper bits:
```
seed = root + k · 2^ROOT_BITS     where k ∈ [0, 2^UPPER_BITS)
```

**Theorem**: All valid k values satisfy `k ≡ k₀ (mod 2^14)` for some offset k₀.

In other words, valid k's are spaced exactly 16,384 apart.

#### Why Does This Happen?

When k increases by 1, the seed increases by 2^ROOT_BITS = 2^21.

The residue for each chunk changes by:
```
Δᵢ = (2^21 · MULT >> 17) mod 10 = (2^4 · MULT) mod 10 = 2
```

Since Δᵢ = 2 (even), residues can only change by even amounts:
```
k=0:  residues = [4, 8, 6, 6, 4, 0, 6, ...]  (all even!)
k=1:  residues = [6, 2, 8, 0, 8, 2, 8, ...]  (still even!)
k=2:  residues = [8, 4, 0, 2, 2, 4, 0, ...]  (still even!)
```

For residue to go from non-zero to zero, it must traverse: 8→6→4→2→0 or 2→0.
The period of this cycle combined with the 16-constraint system creates a
period of 2^14 = 16,384 in k-space.

#### Visual Example

```
Valid k values for position (325, -533), root 0x1960CF:

k mod 16384:  ████ 7498
              │
              │  All 46 valid k values ≡ 7498 (mod 16384)
              │
              ├── k = 7498        = 7498 + 0·16384
              ├── k = 4087114     = 7498 + 249·16384  
              ├── k = 7757130     = 7498 + 473·16384
              ├── k = 8166730     = 7498 + 498·16384
              └── ... (42 more, all ≡ 7498 mod 16384)

k-space structure:
    0        16384      32768      49152      65536     ...
    │          │          │          │          │
    └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────
         │          │          │          │
       7498      23882      40266      56650    (k₀ + n·16384)
         │                                
         └── Only check these values
```

---

### Phase 3: Strided Enumeration

#### The Algorithm

```python
for each root in hensel_roots:                    # ~7,500 roots
    
    # Phase 2a: Find k-offset (or determine root is "false")
    k_offset = NONE
    for k in range(16384):                        # Scan [0, 16384)
        seed = root + k * 2^ROOT_BITS
        if is_valid_4x4(seed):
            k_offset = k
            break
    
    if k_offset == NONE:
        continue  # This root has no valid seeds (Hensel false positive)
    
    # Phase 3: Enumerate with stride
    for n in range(8192):                         # 2^27 / 2^14 = 8,192
        k = k_offset + n * 16384
        seed = root + k * 2^ROOT_BITS
        if is_valid_4x4(seed):
            output(seed)
```

#### Average Compute

```
┌──────────────────────┬─────────────────────┬───────────────────────┐
│       Phase          │      Work           │      For 4×4          │
├──────────────────────┼─────────────────────┼───────────────────────┤
│ Hensel Lifting       │ O(2^ROOT_BITS)      │ ~2,000,000            │
│ K-Offset Discovery   │ O(roots × 16384)    │ ~123,000,000          │
│ Strided Enumeration  │ O(roots × 8192)     │ ~61,000,000           │
├──────────────────────┼─────────────────────┼───────────────────────┤
│ TOTAL                │                     │ ~184,000,000          │
└──────────────────────┴─────────────────────┴───────────────────────┘

Compare k offset stride, to normal enumeration: 7,500 × 134,217,728 = 1,006,632,960,000

```

---

### The Lattice Structure of Valid Seeds

Valid seeds don't just satisfy `k ≡ k₀ (mod 16384)`. They form a 2D lattice.

#### Observed Structure

For position (325, -533):
```
Valid k values (normalized by subtracting first):

k_relative = 0, 4079616, 7749632, 8159232, 11829248, ...

Divide by 16384:

m = 0, 249, 473, 498, 722, 747, 971, 996, 1220, 1245, ...

Express as 25a + 224b:

m =   0 = 25·0  + 224·0
m = 249 = 25·1  + 224·1
m = 473 = 25·1  + 224·2
m = 498 = 25·2  + 224·2
m = 722 = 25·2  + 224·3
    ...
```

The valid m values form a numerical semigroup generated by {25, 224}:
```
Valid m = { 25a + 224b : a,b ≥ 0 }
```

#### Lattice Visualization

```
        b (multiples of 224)
        │
     8  │  ·  ·  ·  ·  ·  ·  ·  ·  ·
        │
     7  │  ·  ·  ·  ·  ·  ·  ·  ·  ·
        │
     6  │  ·  ·  ·  ·  ·  ·  ·  ·  ·
        │
     5  │  ·  ·  ·  ·  ·  ★  ·  ·  ·   ← (a=4, b=5): m = 100+1120 = 1220
        │
     4  │  ·  ·  ·  ★  ★  ·  ·  ·  ·   ← (3,4)=971, (4,4)=996
        │
     3  │  ·  ·  ★  ★  ·  ·  ·  ·  ·   ← (2,3)=722, (3,3)=747
        │
     2  │  ·  ★  ★  ·  ·  ·  ·  ·  ·   ← (1,2)=473, (2,2)=498
        │
     1  │  ★  ·  ·  ·  ·  ·  ·  ·  ·   ← (1,1)=249
        │
     0  │  ★  ·  ·  ·  ·  ·  ·  ·  ·   ← (0,0)=0 (the k-offset)
        └──┴──┴──┴──┴──┴──┴──┴──┴──────▶ a (multiples of 25)
           0  1  2  3  4  5  6  7  8
```

This means valid k's are even sparser than just "every 16384th value".

---

### Pattern Size Scaling

The constants adapt to pattern size:

```
┌─────────┬───────────┬─────────────┬───────────┬───────────────┐
│ Pattern │ ROOT_BITS │ UPPER_BITS  │ K_STRIDE  │ Speedup       │
├─────────┼───────────┼─────────────┼───────────┼───────────────┤
│   3×3   │    19     │     29      │   4,096   │    4,096×     │
│   4×4   │    21     │     27      │  16,384   │   16,384×     │
│   5×5   │    24     │     24      │ 131,072   │  131,072×     │
│   6×6   │    26     │     22      │ 524,288   │  524,288×     │
│   7×7   │    28     │     20      │ 1,048,576 │ 1,048,576×    │
└─────────┴───────────┴─────────────┴───────────┴───────────────┘

Formula: K_STRIDE = 2^(ROOT_BITS - 7)
```

Larger patterns have more constraints, which:
1. Lock more lower bits (higher ROOT_BITS)
2. Create stronger periodicity (larger K_STRIDE)
3. Results in dramatically faster searches

---

### Output

Results written to `slime_NxN_results.txt`:
```
# 4x4 Slime Chunk Seeds for position (325, -533)
# Format: seed
15726108879
8571300962511
16267882356943
...
```

Progress logged to `slime_current_position.txt` for resumption.

---

## slime_fft

Finds all seeds matching a custom pattern loaded from pattern.txt.

```
slimefft.exe [--test]
```

Supports wildcards (X) and anti-patterns (.) unlike slime.exe.

### Algorithm: FFT Correlation

For each seed, finding pattern matches across millions of positions is expensive.
FFT convolution finds ALL matches in O(N log N) instead of O(N * pattern_size).

1. Generate seeds using LCG math (only seeds where first chunk could be slime)
2. For each seed batch, generate 4096x4096 tile bitmap (slime=1, not=0)
3. FFT the bitmap and pattern kernel
4. Multiply in frequency domain (convolution theorem)
5. Inverse FFT - peaks indicate pattern matches
6. If pattern has anti-cells (.), run second pass on inverted bitmap

Processes ~16 seeds in parallel using multiple FFT slots.
Tiles spiral outward from origin.

Output: `slime_results.txt`

---

## slime_world

Finds all positions matching a pattern for ONE specific seed.

```
slime_world.exe <seed> [--radius R] [--pattern FILE] [--size N] [--flush N]
```

Options:
- `--radius R` - search radius in chunks (default 100000)
- `--pattern FILE` - load pattern from file
- `--size N` - use solid NxN pattern
- `--flush N` - write results every N seconds (default 10)

### Algorithm: GPU Brute Force with First-Cell Filter

No fancy math here - just raw GPU throughput. But filtering helps.

1. Load pattern into GPU constant memory
2. For each candidate position, check first required slime cell (~10% pass)
3. If pass: check remaining pattern cells (early exit on first fail)
4. Write matches to results file periodically

Why not FFT? FFT helps when searching many seeds against fixed positions.
Here we have one seed and billions of positions - generating the full bitmap
would explode memory. Brute force with early-exit is actually optimal.

Achieves ~300 billion positions/second on an RTX 4080 Super.

Output: `slime_world_<seed>_<pattern>.txt`

---

## slime_refiner

Post-processes results from slime or slime_fft. Two modes:

```
slime_refiner.exe [--rect|--glob] [--pattern N] [--max N] [input] [output]
```

### Rect Mode (default)

Grows each NxN pattern into the largest possible rectangle.

1. Start with found pattern location
2. Try expand in each direction (check full row/column)
3. Repeat until stuck
4. Report final dimensions

### Glob Mode

Finds largest contiguous slime region (any shape) via flood fill.

1. Start at pattern location
2. BFS flood fill to find all connected slime chunks
3. Track bounding box and total chunk count
4. Find largest inscribed rectangle within the glob
5. Report glob size, density, best rect

Works with both slime_results.txt and slime_world output formats.

Output: `refined_results.txt` (rect) or `glob_results.txt` (glob)




