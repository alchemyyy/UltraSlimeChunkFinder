/**
 * Optimized Hensel Lifting Slime Chunk Finder
 *
 * KEY INSIGHT FROM MATHEMATICAL ANALYSIS:
 * For a 4x4 slime chunk pattern, the 16 simultaneous congruences effectively
 * "lock" the lower ~18-20 bits of the seed into specific root values.
 *
 * OPTIMIZATION STRATEGY:
 * 1. Use Hensel lifting to find ALL valid lower-bit roots (mod 2^20)
 * 2. For each root, only enumerate upper bits (28 bits instead of 48)
 * 3. Seeds form arithmetic progressions: valid seeds differ by 2^k
 *
 * This reduces search from 2^48 to ~2^28 per position = 10^8x speedup
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <set>
#include <map>

// ============================================================================
// LCG Constants
// ============================================================================

constexpr uint64_t MASK_48 = (1ULL << 48) - 1;
constexpr uint64_t LCG_MULT = 0x5DEECE66DULL;
constexpr uint64_t LCG_ADD = 0xBULL;
constexpr uint64_t XOR_CONST = 0x3ad8025fULL;
constexpr uint64_t COMBINED_XOR = XOR_CONST ^ LCG_MULT;

// Slime formula constants
constexpr int32_t SLIME_A = 0x4c1906;
constexpr int32_t SLIME_B = 0x5ac0db;
constexpr int64_t SLIME_C = 0x4307a7LL;
constexpr int32_t SLIME_D = 0x5f24f;

// Pattern size - use #define for preprocessor conditionals
#define PATTERN_SIZE 5
constexpr int NUM_CONSTRAINTS = PATTERN_SIZE * PATTERN_SIZE;

// ============================================================================
// OPTIMIZED HENSEL PARAMETERS
// ============================================================================

// ROOT_BITS: Number of lower bits to solve via Hensel lifting
// More constraints = more bits locked by the system of congruences
//
// Mathematical basis:
// - Each constraint (nextInt(10) == 0) provides ~3.32 bits of information (log2(10))
// - But constraints share lower bits, so effective bits locked ≈ 17 + log2(constraints)
//
// Recommended values:
//   3x3 ( 9 constraints): ROOT_BITS = 18-19, locks ~17 + 3.2 = ~20 bits
//   4x4 (16 constraints): ROOT_BITS = 20-21, locks ~17 + 4.0 = ~21 bits
//   5x5 (25 constraints): ROOT_BITS = 22-24, locks ~17 + 4.6 = ~22 bits
//   6x6 (36 constraints): ROOT_BITS = 24-26, locks ~17 + 5.2 = ~22 bits
//   7x7 (49 constraints): ROOT_BITS = 26-28, locks ~17 + 5.6 = ~23 bits
//
// Higher ROOT_BITS = more Hensel iterations but fewer upper bits to enumerate
// Sweet spot: set ROOT_BITS slightly above the theoretical lock point

#if PATTERN_SIZE == 3
    constexpr int ROOT_BITS = 19;
#elif PATTERN_SIZE == 4
    constexpr int ROOT_BITS = 21;
#elif PATTERN_SIZE == 5
    constexpr int ROOT_BITS = 24;
#elif PATTERN_SIZE == 6
    constexpr int ROOT_BITS = 26;
#elif PATTERN_SIZE == 7
    constexpr int ROOT_BITS = 28;
#else
    constexpr int ROOT_BITS = 20;  // Default fallback
#endif

// After finding roots, we only need to enumerate upper bits
constexpr int UPPER_BITS = 48 - ROOT_BITS;
constexpr uint64_t UPPER_COUNT = 1ULL << UPPER_BITS;

// For Hensel lifting phase
constexpr int HENSEL_START_BIT = 18;  // First bit affecting output

// Two-phase Hensel: refine roots from ROOT_BITS to REFINE_BITS before enumeration
// This filters out dead-end roots that pass initial Hensel but can't extend further
constexpr int REFINE_BITS = 32;  // Lift roots to this many bits to filter dead ends

// ============================================================================
// K-SPACE STRIDE OPTIMIZATION
// ============================================================================
// Mathematical discovery: For any position, valid k values (upper bits) satisfy:
//   k ≡ offset (mod 2^K_STRIDE_BITS) for some offset in [0, K_STRIDE)
//
// This is because residue deltas when incrementing k by 1 are always even
// (due to bit alignment in the LCG formula), and the mod-5 constraint system
// has period 2^(ROOT_BITS - 7) in k-space.
//
// Formula: K_STRIDE_BITS = ROOT_BITS - 7
//
// This allows a K_STRIDE-fold speedup:
// - Phase 1: Scan k ∈ [0, K_STRIDE) to find valid offset (or determine root is false)
// - Phase 2: Enumerate k = offset + n*K_STRIDE for all valid n
constexpr int K_STRIDE_BITS = ROOT_BITS - 7;  // Derived from ROOT_BITS
constexpr uint64_t K_STRIDE = 1ULL << K_STRIDE_BITS;
constexpr uint64_t K_STRIDE_COUNT = UPPER_COUNT / K_STRIDE;

constexpr uint32_t MAX_ROOTS = 1U << 24;  // 16M max roots (realistically <<1000, but costs only 128MB)
constexpr uint32_t MAX_CANDIDATES = 1U << 24;  // 16M for intermediate Hensel steps
constexpr uint32_t MAX_RESULTS = 1U << 20;
constexpr int32_t MAX_POSITION_RADIUS = 1875000;

// ============================================================================
// Hensel Achievability Table
// ============================================================================

__constant__ uint16_t d_achievableMask[32];
uint16_t h_achievableMask[32];

void initHenselTables() {
    for (int r = 0; r < 32; r++) {
        uint16_t mask = 0;
        if (r == 0) {
            mask = 0x3FF;
        } else {
            uint64_t pow2mod10 = (1ULL << r) % 10;
            for (int startRes = 0; startRes < 10; startRes++) {
                for (int k = 0; k < 10; k++) {
                    if ((startRes + k * pow2mod10) % 10 == 0) {
                        mask |= (1 << startRes);
                        break;
                    }
                }
            }
        }
        h_achievableMask[r] = mask;
    }
    cudaMemcpyToSymbol(d_achievableMask, h_achievableMask, sizeof(h_achievableMask));
}

// ============================================================================
// Position term computation
// ============================================================================

__host__ __device__ __forceinline__
int64_t computePositionTerm(int32_t x, int32_t z) {
    // this old one breaks overflows. Makes for some cool partial detections though!
    // int32_t xTerm = x * (x * SLIME_A + SLIME_B);
    // int64_t zTerm = (int64_t)(z * z) * SLIME_C + z * SLIME_D;
    // return (int64_t)xTerm + zTerm;
    // Must match Java's overflow behavior exactly:
    // seed + (long)(x * x * A) + (long)(x * B) + (long)(z * z) * C + (long)(z * D) ^ XOR
    int32_t xxA = x * x * SLIME_A;  // int32 overflow
    int32_t xB = x * SLIME_B;       // int32 overflow
    int32_t zz = z * z;             // int32 overflow
    int32_t zD = z * SLIME_D;       // int32 overflow
    return (int64_t)xxA + (int64_t)xB + (int64_t)zz * SLIME_C + (int64_t)zD;
}

// ============================================================================
// Device: Full slime check (for verification)
// ============================================================================

__device__ __forceinline__
bool isSlimeChunk(uint64_t worldSeed, int32_t x, int32_t z) {
    int64_t posTerm = computePositionTerm(x, z);
    int64_t slimeSeed = (int64_t)worldSeed + posTerm;
    uint64_t internal = ((uint64_t)slimeSeed ^ COMBINED_XOR) & MASK_48;
    uint64_t advanced = (internal * LCG_MULT + LCG_ADD) & MASK_48;
    return (advanced >> 17) % 10 == 0;
}

__device__ __forceinline__
bool checkPattern(uint64_t worldSeed, int32_t baseX, int32_t baseZ) {
    #pragma unroll
    for (int dz = 0; dz < PATTERN_SIZE; dz++) {
        #pragma unroll
        for (int dx = 0; dx < PATTERN_SIZE; dx++) {
            if (!isSlimeChunk(worldSeed, baseX + dx, baseZ + dz)) {
                return false;
            }
        }
    }
    return true;
}

// ============================================================================
// Device: Hensel constraint check at specific bit level
// ============================================================================

__device__ __forceinline__
bool canSatisfyConstraintAtBit(uint64_t partialSeed, int k, int64_t posTerm) {
    if (k <= 17) return true;

    uint64_t mask = (1ULL << k) - 1;
    int64_t partialSum = (int64_t)(partialSeed & mask) + posTerm;
    uint64_t partialInternal = ((uint64_t)partialSum ^ COMBINED_XOR) & mask;
    uint64_t partialLCG = (partialInternal * LCG_MULT + LCG_ADD) & mask;

    int outputBits = k - 17;

    if (outputBits >= 31) {
        uint64_t fullOutput = (partialLCG >> 17) & 0x7FFFFFFF;
        return (fullOutput % 10) == 0;
    }

    uint64_t knownOutput = (partialLCG >> 17) & ((1ULL << outputBits) - 1);
    uint64_t residue = knownOutput % 10;
    uint16_t achievable = d_achievableMask[outputBits];

    return (achievable & (1 << residue)) != 0;
}

__device__ __forceinline__
bool canSatisfyAllConstraints(uint64_t partialSeed, int k, const int64_t* posTerms) {
    #pragma unroll
    for (int i = 0; i < NUM_CONSTRAINTS; i++) {
        if (!canSatisfyConstraintAtBit(partialSeed, k, posTerms[i])) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Kernel: Initialize candidates at HENSEL_START_BIT
// ============================================================================

__global__ void initCandidates(
    const int64_t* __restrict__ posTerms,
    uint64_t* candidates,
    uint32_t* candidateCount,
    uint32_t maxCandidates
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t totalCandidates = 1U << HENSEL_START_BIT;

    if (idx >= totalCandidates) return;

    uint64_t candidate = idx;

    if (canSatisfyAllConstraints(candidate, HENSEL_START_BIT, posTerms)) {
        uint32_t pos = atomicAdd(candidateCount, 1);
        if (pos < maxCandidates) {
            candidates[pos] = candidate;
        }
    }
}

// ============================================================================
// Kernel: Expand and filter (Hensel lifting one bit at a time)
// ============================================================================

__global__ void expandAndFilter(
    const uint64_t* __restrict__ inCandidates,
    uint32_t numIn,
    int currentBit,
    const int64_t* __restrict__ posTerms,
    uint64_t* outCandidates,
    uint32_t* outCount,
    uint32_t maxCandidates
) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;

    uint32_t candidateIdx = idx >> 1;
    uint32_t bitValue = idx & 1;

    if (candidateIdx >= numIn) return;

    uint64_t candidate = inCandidates[candidateIdx];
    uint64_t newCandidate = candidate | ((uint64_t)bitValue << currentBit);

    int newBits = currentBit + 1;

    if (canSatisfyAllConstraints(newCandidate, newBits, posTerms)) {
        uint32_t pos = atomicAdd(outCount, 1);
        if (pos < maxCandidates) {
            outCandidates[pos] = newCandidate;
        }
    }
}

// ============================================================================
// Kernel: Enumerate upper bits and verify
// Given roots (lower ROOT_BITS), enumerate all upper bit combinations
// ============================================================================

__global__ void enumerateUpperBits(
    const uint64_t* __restrict__ roots,
    uint32_t numRoots,
    int32_t baseX, int32_t baseZ,
    uint64_t* results,
    uint32_t* resultCount,
    uint32_t maxResults,
    uint64_t upperStart,
    uint64_t upperEnd
) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t totalWork = (uint64_t)numRoots * (upperEnd - upperStart);

    if (idx >= totalWork) return;

    uint32_t rootIdx = idx % numRoots;
    uint64_t upperIdx = upperStart + (idx / numRoots);

    uint64_t root = roots[rootIdx];
    uint64_t seed = root | (upperIdx << ROOT_BITS);

    // Full verification
    if (checkPattern(seed, baseX, baseZ)) {
        uint32_t pos = atomicAdd(resultCount, 1);
        if (pos < maxResults) {
            results[pos] = seed;
        }
    }
}

// ============================================================================
// Kernel: Generate seeds using discovered stride
// ============================================================================

__global__ void generateSeedsWithStride(
    const uint64_t* __restrict__ baseSeeds,
    uint32_t numBaseSeeds,
    uint64_t stride,
    uint64_t maxSeed,
    int32_t baseX, int32_t baseZ,
    uint64_t* results,
    uint32_t* resultCount,
    uint32_t maxResults,
    uint64_t workOffset
) {
    uint64_t idx = workOffset + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint32_t baseIdx = idx % numBaseSeeds;
    uint64_t strideMultiple = idx / numBaseSeeds;

    uint64_t seed = baseSeeds[baseIdx] + strideMultiple * stride;

    if (seed > maxSeed) return;

    if (checkPattern(seed, baseX, baseZ)) {
        uint32_t pos = atomicAdd(resultCount, 1);
        if (pos < maxResults) {
            results[pos] = seed;
        }
    }
}

// ============================================================================
// Kernel: Batch process upper bits with coalesced memory access
// ============================================================================

__global__ void enumerateUpperBitsBatched(
    const uint64_t* __restrict__ roots,
    uint32_t numRoots,
    int32_t baseX, int32_t baseZ,
    const int64_t* __restrict__ posTerms,
    uint64_t* results,
    uint32_t* resultCount,
    uint32_t maxResults,
    uint64_t upperStart,
    uint64_t upperCount
) {
    uint32_t rootIdx = blockIdx.y;
    if (rootIdx >= numRoots) return;

    uint64_t root = roots[rootIdx];

    uint64_t upperIdx = upperStart + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;

    if (upperIdx >= upperStart + upperCount) return;

    uint64_t seed = root | (upperIdx << ROOT_BITS);

    if (checkPattern(seed, baseX, baseZ)) {
        uint32_t pos = atomicAdd(resultCount, 1);
        if (pos < maxResults) {
            results[pos] = seed;
        }
    }
}

// ============================================================================
// Kernel: Probe roots using known stride
// Each thread checks one root at a few stride multiples to see if it has any seeds
// ============================================================================
// Kernel: Two-phase k-stride enumeration
// Phase 1: Find k-offset for each root by scanning [0, K_STRIDE)
// Phase 2: Enumerate k = offset + n*K_STRIDE
// ============================================================================

// Phase 1: Find the k-offset for a root (if any valid k exists in [0, K_STRIDE))
// Each thread handles one (root, k) pair
__global__ void findKOffsetKernel(
    const uint64_t* __restrict__ roots,
    uint32_t numRoots,
    int32_t baseX, int32_t baseZ,
    uint64_t* kOffsets,      // Output: first valid k for each root (or UINT64_MAX if none)
    uint32_t* hasValidK      // Output: 1 if valid k found, 0 otherwise
) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t totalWork = (uint64_t)numRoots * K_STRIDE;

    if (idx >= totalWork) return;

    uint32_t rootIdx = idx / K_STRIDE;
    uint64_t k = idx % K_STRIDE;

    uint64_t root = roots[rootIdx];
    uint64_t seed = root | (k << ROOT_BITS);

    if (checkPattern(seed, baseX, baseZ)) {
        // Found a valid k! Use atomicMin to keep the smallest
        atomicMin((unsigned long long*)&kOffsets[rootIdx], (unsigned long long)k);
        hasValidK[rootIdx] = 1;
    }
}

// Phase 2: Enumerate all valid seeds using the k-offset and stride
// Each thread handles one (root, stride_multiple) pair
__global__ void enumerateWithKStrideKernel(
    const uint64_t* __restrict__ roots,
    const uint64_t* __restrict__ kOffsets,
    const uint32_t* __restrict__ hasValidK,
    uint32_t numRoots,
    int32_t baseX, int32_t baseZ,
    uint64_t* results,
    uint32_t* resultCount,
    uint32_t maxResults
) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t totalWork = (uint64_t)numRoots * K_STRIDE_COUNT;

    if (idx >= totalWork) return;

    uint32_t rootIdx = idx / K_STRIDE_COUNT;
    uint64_t strideMultiple = idx % K_STRIDE_COUNT;

    // Skip roots that have no valid k
    if (!hasValidK[rootIdx]) return;

    uint64_t root = roots[rootIdx];
    uint64_t kOffset = kOffsets[rootIdx];

    // k = kOffset + strideMultiple * K_STRIDE
    uint64_t k = kOffset + strideMultiple * K_STRIDE;

    // Check bounds
    if (k >= UPPER_COUNT) return;

    uint64_t seed = root | (k << ROOT_BITS);

    if (checkPattern(seed, baseX, baseZ)) {
        uint32_t pos = atomicAdd(resultCount, 1);
        if (pos < maxResults) {
            results[pos] = seed;
        }
    }
}

// ============================================================================

__global__ void probeRootsWithStrideKernel(
    const uint64_t* __restrict__ roots,
    uint32_t numRoots,
    int32_t baseX, int32_t baseZ,
    uint64_t stride,
    uint32_t maxMultiples,  // check multiples 0, 1, 2, ... maxMultiples-1
    uint32_t* hasSeeds  // set to 1 if seed found
) {
    uint32_t rootIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rootIdx >= numRoots) return;

    uint64_t root = roots[rootIdx];

    // Check if this root produces any seed at stride multiples
    for (uint32_t k = 0; k < maxMultiples; k++) {
        uint64_t seed = root + k * stride;
        if (seed > MASK_48) break;

        if (checkPattern(seed, baseX, baseZ)) {
            hasSeeds[rootIdx] = 1;
            return;
        }
    }

    hasSeeds[rootIdx] = 0;
}

// ============================================================================
// Kernel: Refine roots by attempting to lift from ROOT_BITS to REFINE_BITS
// Each thread handles one root and outputs 1 if it survives, 0 if dead end
// ============================================================================

__global__ void refineRootsKernel(
    const uint64_t* __restrict__ roots,
    uint32_t numRoots,
    const int64_t* __restrict__ posTerms,
    uint32_t* survivorFlags  // 1 = survives, 0 = dead end
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRoots) return;

    uint64_t root = roots[idx];

    // Try to lift this root from ROOT_BITS to REFINE_BITS
    // We do a mini-Hensel: try all 2^(REFINE_BITS - ROOT_BITS) extensions
    // But that's 2^11 = 2048 attempts per root - too many for one thread
    //
    // Instead, do iterative lifting: at each bit, check if 0 or 1 works
    // If neither works, the root is dead. If at least one works, continue.
    // We only need to know IF any extension survives, not enumerate all of them.

    // Use a small stack-based approach: track up to 64 candidates at each level
    // For efficiency, we'll just check if ANY path survives

    uint64_t candidates[64];
    uint64_t nextCandidates[64];
    int numCandidates = 1;
    candidates[0] = root;

    for (int bit = ROOT_BITS; bit < REFINE_BITS; bit++) {
        int numNext = 0;

        for (int i = 0; i < numCandidates && numNext < 64; i++) {
            uint64_t cand = candidates[i];

            // Try bit = 0
            uint64_t cand0 = cand;  // bit is already 0
            if (canSatisfyAllConstraints(cand0, bit + 1, posTerms)) {
                if (numNext < 64) nextCandidates[numNext++] = cand0;
            }

            // Try bit = 1
            uint64_t cand1 = cand | (1ULL << bit);
            if (canSatisfyAllConstraints(cand1, bit + 1, posTerms)) {
                if (numNext < 64) nextCandidates[numNext++] = cand1;
            }
        }

        if (numNext == 0) {
            // Dead end - no extensions survive
            survivorFlags[idx] = 0;
            return;
        }

        // Copy next to current
        numCandidates = numNext;
        for (int i = 0; i < numCandidates; i++) {
            candidates[i] = nextCandidates[i];
        }
    }

    // At least one extension survived to REFINE_BITS
    survivorFlags[idx] = 1;
}

// ============================================================================
// Kernel: Enumerate a single root's upper bits
// ============================================================================

__global__ void enumerateSingleRoot(
    uint64_t root,
    int32_t baseX, int32_t baseZ,
    uint64_t* results,
    uint32_t* resultCount,
    uint32_t maxResults,
    uint64_t upperStart,
    uint64_t upperCount
) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= upperCount) return;

    uint64_t upperIdx = upperStart + idx;
    uint64_t seed = root | (upperIdx << ROOT_BITS);

    if (checkPattern(seed, baseX, baseZ)) {
        uint32_t pos = atomicAdd(resultCount, 1);
        if (pos < maxResults) {
            results[pos] = seed;
        }
    }
}

// ============================================================================
// Spiral Position Generator
// ============================================================================

struct SpiralIterator {
    int32_t x, z;
    int32_t dx, dz;
    int32_t segmentLength, segmentPassed;

    SpiralIterator() : x(0), z(0), dx(1), dz(0), segmentLength(1), segmentPassed(0) {}

    void next() {
        x += dx;
        z += dz;
        segmentPassed++;
        if (segmentPassed == segmentLength) {
            segmentPassed = 0;
            int32_t temp = dx;
            dx = -dz;
            dz = temp;
            if (dz == 0) segmentLength++;
        }
    }

    int32_t radius() const { return std::max(std::abs(x), std::abs(z)); }
};

// ============================================================================
// Host verification (forward declarations)
// ============================================================================

bool hostIsSlime(int64_t seed, int32_t x, int32_t z);
bool hostCheckPattern(int64_t seed, int32_t baseX, int32_t baseZ);

// ============================================================================
// Optimized Hensel Slime Searcher
// ============================================================================

class OptimizedHenselSlimeSearcher {
private:
    // GPU resources
    int64_t* d_posTerms;
    uint64_t* d_candidates[2];
    uint32_t* d_candidateCount;
    uint64_t* d_roots;
    uint32_t* d_rootCount;
    uint64_t* d_results;
    uint32_t* d_resultCount;
    uint32_t* d_survivorFlags;  // For root refinement
    uint64_t* d_kOffsets;       // For k-stride optimization
    uint32_t* d_hasValidK;      // For k-stride optimization

    // Host resources
    std::vector<int64_t> h_posTerms;
    std::ofstream outFile;
    std::ofstream positionLog;

    uint64_t totalPositionsProcessed;
    uint64_t totalResultsFound;

public:
    OptimizedHenselSlimeSearcher() : totalPositionsProcessed(0), totalResultsFound(0) {
        initHenselTables();

        h_posTerms.resize(NUM_CONSTRAINTS);

        cudaMalloc(&d_posTerms, NUM_CONSTRAINTS * sizeof(int64_t));
        cudaMalloc(&d_candidates[0], MAX_CANDIDATES * sizeof(uint64_t));
        cudaMalloc(&d_candidates[1], MAX_CANDIDATES * sizeof(uint64_t));
        cudaMalloc(&d_candidateCount, sizeof(uint32_t));
        cudaMalloc(&d_roots, MAX_ROOTS * sizeof(uint64_t));
        cudaMalloc(&d_rootCount, sizeof(uint32_t));
        cudaMalloc(&d_results, MAX_RESULTS * sizeof(uint64_t));
        cudaMalloc(&d_resultCount, sizeof(uint32_t));
        cudaMalloc(&d_survivorFlags, MAX_ROOTS * sizeof(uint32_t));
        cudaMalloc(&d_kOffsets, MAX_ROOTS * sizeof(uint64_t));
        cudaMalloc(&d_hasValidK, MAX_ROOTS * sizeof(uint32_t));

        char filename[64];
        snprintf(filename, sizeof(filename), "slime_%dx%d_results.txt", PATTERN_SIZE, PATTERN_SIZE);
        outFile.open(filename, std::ios::app);
        outFile << "# " << PATTERN_SIZE << "x" << PATTERN_SIZE << " Slime Pattern Results (Optimized Hensel)\n";
        outFile << "# Format: baseX,baseZ,seed\n";
        outFile.flush();

        positionLog.open("slime_current_position.txt", std::ios::out);
    }

    ~OptimizedHenselSlimeSearcher() {
        outFile.close();
        positionLog.close();
        cudaFree(d_posTerms);
        cudaFree(d_candidates[0]);
        cudaFree(d_candidates[1]);
        cudaFree(d_candidateCount);
        cudaFree(d_roots);
        cudaFree(d_rootCount);
        cudaFree(d_results);
        cudaFree(d_resultCount);
        cudaFree(d_survivorFlags);
        cudaFree(d_kOffsets);
        cudaFree(d_hasValidK);
    }

    void computePositionTerms(int32_t baseX, int32_t baseZ) {
        for (int dz = 0; dz < PATTERN_SIZE; dz++) {
            for (int dx = 0; dx < PATTERN_SIZE; dx++) {
                h_posTerms[dz * PATTERN_SIZE + dx] = computePositionTerm(baseX + dx, baseZ + dz);
            }
        }
        cudaMemcpy(d_posTerms, h_posTerms.data(), NUM_CONSTRAINTS * sizeof(int64_t), cudaMemcpyHostToDevice);
    }

    // Phase 1: Find all valid roots (lower ROOT_BITS) via Hensel lifting
    uint32_t findRoots() {
        int threads = 256;
        int currentBuffer = 0;

        // Initialize at HENSEL_START_BIT
        cudaMemset(d_candidateCount, 0, sizeof(uint32_t));
        uint32_t initCount = 1U << HENSEL_START_BIT;
        int initBlocks = (initCount + threads - 1) / threads;

        initCandidates<<<initBlocks, threads>>>(
            d_posTerms,
            d_candidates[currentBuffer],
            d_candidateCount,
            MAX_CANDIDATES
        );
        cudaDeviceSynchronize();

        uint32_t numCandidates;
        cudaMemcpy(&numCandidates, d_candidateCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        if (numCandidates == 0) return 0;
        numCandidates = std::min(numCandidates, MAX_CANDIDATES);

        // Lift through remaining bits up to ROOT_BITS
        for (int bit = HENSEL_START_BIT; bit < ROOT_BITS; bit++) {
            int nextBuffer = 1 - currentBuffer;
            cudaMemset(d_candidateCount, 0, sizeof(uint32_t));

            uint64_t totalWork = (uint64_t)numCandidates * 2;
            int blocks = (totalWork + threads - 1) / threads;
            blocks = std::min(blocks, 65535 * 256);

            expandAndFilter<<<blocks, threads>>>(
                d_candidates[currentBuffer],
                numCandidates,
                bit,
                d_posTerms,
                d_candidates[nextBuffer],
                d_candidateCount,
                MAX_CANDIDATES
            );
            cudaDeviceSynchronize();

            cudaMemcpy(&numCandidates, d_candidateCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

            if (numCandidates == 0) return 0;
            numCandidates = std::min(numCandidates, MAX_CANDIDATES);

            currentBuffer = nextBuffer;
        }

        // Copy final roots
        numCandidates = std::min(numCandidates, MAX_ROOTS);
        cudaMemcpy(d_roots, d_candidates[currentBuffer], numCandidates * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

        return numCandidates;
    }

    // Phase 1.5: Refine roots by lifting from ROOT_BITS to REFINE_BITS
    // Returns number of surviving roots (compacts d_roots in place)
    uint32_t refineRoots(uint32_t numRoots) {
        if (numRoots == 0) return 0;

        int threads = 256;
        int blocks = (numRoots + threads - 1) / threads;

        // Launch refinement kernel
        refineRootsKernel<<<blocks, threads>>>(
            d_roots, numRoots, d_posTerms, d_survivorFlags
        );
        cudaDeviceSynchronize();

        // Copy flags to host
        std::vector<uint32_t> h_flags(numRoots);
        cudaMemcpy(h_flags.data(), d_survivorFlags, numRoots * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Copy roots to host
        std::vector<uint64_t> h_roots(numRoots);
        cudaMemcpy(h_roots.data(), d_roots, numRoots * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        // Compact survivors
        std::vector<uint64_t> survivors;
        for (uint32_t i = 0; i < numRoots; i++) {
            if (h_flags[i]) {
                survivors.push_back(h_roots[i]);
            }
        }

        // Copy back to device
        if (!survivors.empty()) {
            cudaMemcpy(d_roots, survivors.data(), survivors.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
        }

        return (uint32_t)survivors.size();
    }

    // Filter roots using known stride - only keep roots that produce seeds
    // Returns number of surviving roots (compacts d_roots in place)
    uint32_t filterRootsWithStride(uint32_t numRoots, int32_t baseX, int32_t baseZ, uint64_t stride) {
        if (numRoots == 0 || stride == 0) return numRoots;

        int threads = 256;
        int blocks = (numRoots + threads - 1) / threads;

        // How many stride multiples to check per root
        // stride * maxMultiples should cover reasonable seed range
        uint32_t maxMultiples = std::min((uint64_t)1000, (MASK_48 / stride) + 1);

        // Launch probe kernel
        probeRootsWithStrideKernel<<<blocks, threads>>>(
            d_roots, numRoots, baseX, baseZ, stride, maxMultiples, d_survivorFlags
        );
        cudaDeviceSynchronize();

        // Copy flags to host
        std::vector<uint32_t> h_flags(numRoots);
        cudaMemcpy(h_flags.data(), d_survivorFlags, numRoots * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Copy roots to host
        std::vector<uint64_t> h_roots(numRoots);
        cudaMemcpy(h_roots.data(), d_roots, numRoots * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        // Compact survivors (roots that found at least one seed)
        std::vector<uint64_t> survivors;
        for (uint32_t i = 0; i < numRoots; i++) {
            if (h_flags[i]) {
                survivors.push_back(h_roots[i]);
            }
        }

        // Copy back to device
        if (!survivors.empty()) {
            cudaMemcpy(d_roots, survivors.data(), survivors.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
        }

        return (uint32_t)survivors.size();
    }

    static uint64_t gcd(uint64_t a, uint64_t b) {
        while (b != 0) { uint64_t t = b; b = a % b; a = t; }
        return a;
    }

    // ========================================================================
    // OPTIMIZED: Two-phase k-stride enumeration
    // Uses the mathematical property that valid k ≡ offset (mod K_STRIDE)
    // This gives ~K_STRIDE-fold speedup over brute force enumeration
    // ========================================================================
    uint64_t searchPositionOptimized(int32_t baseX, int32_t baseZ) {
        computePositionTerms(baseX, baseZ);

        uint32_t numRoots = findRoots();
        if (numRoots == 0) return 0;

        int threads = 256;

        // Initialize k-offsets to UINT64_MAX (meaning "not found")
        // and hasValidK to 0
        cudaMemset(d_hasValidK, 0, numRoots * sizeof(uint32_t));
        std::vector<uint64_t> initOffsets(numRoots, UINT64_MAX);
        cudaMemcpy(d_kOffsets, initOffsets.data(), numRoots * sizeof(uint64_t), cudaMemcpyHostToDevice);

        // ========================================
        // PHASE 1: Find k-offset for each root
        // Scan k ∈ [0, K_STRIDE) to find first valid k per root
        // ========================================
        {
            uint64_t totalWork = (uint64_t)numRoots * K_STRIDE;
            uint64_t blocks = (totalWork + threads - 1) / threads;

            // Limit blocks to avoid launch failure
            constexpr uint64_t MAX_BLOCKS = 1ULL << 24;
            if (blocks > MAX_BLOCKS) {
                // Process in batches
                for (uint64_t start = 0; start < totalWork; start += MAX_BLOCKS * threads) {
                    uint64_t batchWork = std::min((uint64_t)(MAX_BLOCKS * threads), totalWork - start);
                    uint64_t batchBlocks = (batchWork + threads - 1) / threads;

                    // Note: Need a modified kernel that takes a work offset
                    // For simplicity, use single launch if possible
                }
            }

            findKOffsetKernel<<<(uint32_t)std::min(blocks, MAX_BLOCKS), threads>>>(
                d_roots, numRoots, baseX, baseZ,
                d_kOffsets, d_hasValidK
            );
            cudaDeviceSynchronize();
        }

        // Check how many roots have valid k
        std::vector<uint32_t> h_hasValidK(numRoots);
        cudaMemcpy(h_hasValidK.data(), d_hasValidK, numRoots * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        uint32_t numValidRoots = 0;
        for (uint32_t i = 0; i < numRoots; i++) {
            if (h_hasValidK[i]) numValidRoots++;
        }

        if (numValidRoots == 0) {
            return 0;  // No valid seeds for this position
        }

        // ========================================
        // PHASE 2: Enumerate all valid seeds using k-stride
        // For each root with valid k, enumerate k = offset + n*K_STRIDE
        // ========================================
        cudaMemset(d_resultCount, 0, sizeof(uint32_t));

        {
            uint64_t totalWork = (uint64_t)numRoots * K_STRIDE_COUNT;
            uint64_t blocks = (totalWork + threads - 1) / threads;
            constexpr uint64_t MAX_BLOCKS = 1ULL << 24;

            enumerateWithKStrideKernel<<<(uint32_t)std::min(blocks, MAX_BLOCKS), threads>>>(
                d_roots, d_kOffsets, d_hasValidK, numRoots,
                baseX, baseZ,
                d_results, d_resultCount, MAX_RESULTS
            );
            cudaDeviceSynchronize();
        }

        // Collect results
        uint32_t numResults;
        cudaMemcpy(&numResults, d_resultCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        numResults = std::min(numResults, MAX_RESULTS);

        if (numResults == 0) {
            return 0;
        }

        std::vector<uint64_t> results(numResults);
        cudaMemcpy(results.data(), d_results, numResults * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        // Sort and deduplicate
        std::set<uint64_t> uniqueSeeds(results.begin(), results.end());

        // Write results
        for (uint64_t seed : uniqueSeeds) {
            outFile << baseX << "," << baseZ << "," << seed << "\n";
        }
        outFile.flush();

        return uniqueSeeds.size();
    }

    // Phase 2: Enumerate upper bits with stride optimization (LEGACY - kept for comparison)
    uint64_t searchPosition(int32_t baseX, int32_t baseZ) {
        computePositionTerms(baseX, baseZ);

        uint32_t numRoots = findRoots();
        //printf("  [%d,%d] findRoots: %u roots\n", baseX, baseZ, numRoots);
        if (numRoots == 0) return 0;

        // Copy roots to host
        std::vector<uint64_t> h_roots(numRoots);
        cudaMemcpy(h_roots.data(), d_roots, numRoots * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        int threads = 256;
        constexpr uint64_t BATCH_SIZE = 1ULL << 24;

        // Track seeds by root
        std::map<uint64_t, std::vector<uint64_t>> seedsByRoot;

        // ========================================
        // PHASE 1: Enumerate until we find two seeds from the same root
        // ========================================

        uint64_t initialStride = 0;
        uint64_t baseSeedForStride = 0;
        uint32_t prevResultCount = 0;
        uint64_t upperEnumerated = 0;  // Track how far we've enumerated

        cudaMemset(d_resultCount, 0, sizeof(uint32_t));

        for (uint64_t upperStart = 0; upperStart < UPPER_COUNT; upperStart += BATCH_SIZE) {
            uint64_t batchCount = std::min(BATCH_SIZE, UPPER_COUNT - upperStart);
            dim3 grid((batchCount + threads - 1) / threads, numRoots);

            enumerateUpperBitsBatched<<<grid, threads>>>(
                d_roots, numRoots, baseX, baseZ, d_posTerms,
                d_results, d_resultCount, MAX_RESULTS,
                upperStart, batchCount
            );
            cudaDeviceSynchronize();

            upperEnumerated = upperStart + batchCount;

            uint32_t numResults;
            cudaMemcpy(&numResults, d_resultCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

            if (numResults > prevResultCount) {
                uint32_t newCount = numResults - prevResultCount;
                std::vector<uint64_t> newSeeds(newCount);
                cudaMemcpy(newSeeds.data(), d_results + prevResultCount, newCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);

                for (uint64_t seed : newSeeds) {
                    uint64_t root = seed & ((1ULL << ROOT_BITS) - 1);
                    seedsByRoot[root].push_back(seed);

                    // Check if we now have 2 seeds from this root
                    if (initialStride == 0 && seedsByRoot[root].size() == 2) {
                        auto& seeds = seedsByRoot[root];
                        std::sort(seeds.begin(), seeds.end());
                        initialStride = seeds[1] - seeds[0];
                        baseSeedForStride = seeds[0];
                    }
                }
                prevResultCount = numResults;
            }

            // Exit as soon as we have an initial stride
            if (initialStride > 0) break;
        }

        // If no stride found, return what we have (0 or 1 seed per root)
        if (initialStride == 0) {
            std::set<uint64_t> allSeeds;
            for (auto& kv : seedsByRoot) {
                for (uint64_t s : kv.second) {
                    allSeeds.insert(s);
                    outFile << baseX << "," << baseZ << "," << s << "\n";
                }
            }
            outFile.flush();
            return allSeeds.size();
        }

        // ========================================
        // PHASE 2: Refine stride using GCD + divisor verification
        // ========================================
        // The GCD of observed differences might still be a multiple of the true stride
        // if we only found seeds at even intervals. We verify by testing divisors.

        uint64_t stride = initialStride;

        // Compute GCD of all differences from the root that gave us initialStride
        uint64_t rootForStride = baseSeedForStride & ((1ULL << ROOT_BITS) - 1);
        auto& seedsForGCD = seedsByRoot[rootForStride];
        if (seedsForGCD.size() >= 2) {
            std::sort(seedsForGCD.begin(), seedsForGCD.end());
            uint64_t g = 0;
            for (size_t i = 1; i < seedsForGCD.size(); i++) {
                g = gcd(g, seedsForGCD[i] - seedsForGCD[i-1]);
            }
            if (g > 0) {
                stride = g;
            }
        }

        // Verify stride by testing if smaller divisors also work
        // The true stride is the smallest divisor d of our computed stride
        // such that baseSeed + d (or baseSeed - d) is a valid seed
        uint64_t baseSeed = seedsForGCD[0];
        std::vector<uint64_t> divisors;
        for (uint64_t d = 1; d * d <= stride; d++) {
            if (stride % d == 0) {
                divisors.push_back(d);
                if (d != stride / d) {
                    divisors.push_back(stride / d);
                }
            }
        }
        std::sort(divisors.begin(), divisors.end());

        for (uint64_t d : divisors) {
            if (d == stride) break;  // Already verified this works

            // Check if base + d is valid
            uint64_t candidateFwd = baseSeed + d;
            if (candidateFwd <= MASK_48 && hostCheckPattern(candidateFwd, baseX, baseZ)) {
                stride = d;
                break;
            }
            // Check if base - d is valid (in case base isn't the first seed)
            if (baseSeed >= d) {
                uint64_t candidateBwd = baseSeed - d;
                if (hostCheckPattern(candidateBwd, baseX, baseZ)) {
                    stride = d;
                    break;
                }
            }
        }

        // ========================================
        // PHASE 2.5: Filter roots using stride
        // ========================================
        // Now that we have the stride, quickly filter out roots that don't produce any seeds

        uint32_t numFilteredRoots = filterRootsWithStride(numRoots, baseX, baseZ, stride);

        // Update h_roots with filtered roots
        if (numFilteredRoots > 0 && numFilteredRoots < numRoots) {
            h_roots.resize(numFilteredRoots);
            cudaMemcpy(h_roots.data(), d_roots, numFilteredRoots * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            numRoots = numFilteredRoots;
        }

        // ========================================
        // PHASE 3: Find one base seed per congruence class
        // ========================================
        // Enumerate one full stride period to find all roots with seeds.
        // We need one representative from each congruence class mod stride.

        uint64_t strideInUpper = stride >> ROOT_BITS;
        uint64_t targetUpper = (strideInUpper > 0) ? strideInUpper : 1;

        // Continue enumeration from where we left off until we cover a full period
        for (uint64_t upperStart = upperEnumerated; upperStart < targetUpper; upperStart += BATCH_SIZE) {
            uint64_t batchCount = std::min(BATCH_SIZE, targetUpper - upperStart);
            dim3 grid((batchCount + threads - 1) / threads, numRoots);

            enumerateUpperBitsBatched<<<grid, threads>>>(
                d_roots, numRoots, baseX, baseZ, d_posTerms,
                d_results, d_resultCount, MAX_RESULTS,
                upperStart, batchCount
            );
            cudaDeviceSynchronize();

            uint32_t numResults;
            cudaMemcpy(&numResults, d_resultCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

            if (numResults > prevResultCount) {
                uint32_t newCount = numResults - prevResultCount;
                std::vector<uint64_t> newSeeds(newCount);
                cudaMemcpy(newSeeds.data(), d_results + prevResultCount, newCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);

                for (uint64_t seed : newSeeds) {
                    uint64_t root = seed & ((1ULL << ROOT_BITS) - 1);
                    seedsByRoot[root].push_back(seed);
                }
                prevResultCount = numResults;
            }
        }

        // Collect base seeds (one per root/congruence class)
        std::set<uint64_t> allFoundSeeds;
        std::vector<uint64_t> baseSeeds;

        for (auto& kv : seedsByRoot) {
            auto& seeds = kv.second;
            std::sort(seeds.begin(), seeds.end());
            baseSeeds.push_back(seeds[0]);
            for (uint64_t s : seeds) {
                allFoundSeeds.insert(s);
            }
        }

        if (allFoundSeeds.empty()) {
            return 0;
        }

        // ========================================
        // PHASE 4: Generate all seeds using stride
        // ========================================

        std::set<uint64_t> allSeeds = allFoundSeeds;  // Start with enumerated seeds

        if (stride > 0 && !baseSeeds.empty()) {
            // Generate all seeds using stride for each root that has a base seed
            cudaMemset(d_resultCount, 0, sizeof(uint32_t));

            uint64_t* d_baseSeeds;
            cudaMalloc(&d_baseSeeds, baseSeeds.size() * sizeof(uint64_t));
            cudaMemcpy(d_baseSeeds, baseSeeds.data(), baseSeeds.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

            uint64_t maxMultiples = (MASK_48 / stride) + 1;
            uint64_t genTotalWork = baseSeeds.size() * maxMultiples;

            constexpr uint64_t GEN_BATCH_SIZE = 1ULL << 24;
            for (uint64_t start = 0; start < genTotalWork; start += GEN_BATCH_SIZE) {
                uint64_t count = std::min(GEN_BATCH_SIZE, genTotalWork - start);
                int blocks = (count + threads - 1) / threads;

                generateSeedsWithStride<<<blocks, threads>>>(
                    d_baseSeeds, (uint32_t)baseSeeds.size(), stride, MASK_48,
                    baseX, baseZ, d_results, d_resultCount, MAX_RESULTS, start
                );
            }
            cudaDeviceSynchronize();
            cudaFree(d_baseSeeds);

            uint32_t genCount;
            cudaMemcpy(&genCount, d_resultCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            genCount = std::min(genCount, MAX_RESULTS);


            if (genCount > 0) {
                std::vector<uint64_t> genSeeds(genCount);
                cudaMemcpy(genSeeds.data(), d_results, genCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);
                for (uint64_t seed : genSeeds) {
                    allSeeds.insert(seed);
                }
            }
        }

        // Write results
        for (uint64_t seed : allSeeds) {
            outFile << baseX << "," << baseZ << "," << seed << "\n";
        }
        outFile.flush();

        return allSeeds.size();
    }

    void search(int32_t maxRadius, int32_t startX = 0, int32_t startZ = 0) {
        bool unlimited = (maxRadius <= 0);

        if (!unlimited && maxRadius > MAX_POSITION_RADIUS) {
            printf("Warning: radius clamped to %d\n", MAX_POSITION_RADIUS);
            maxRadius = MAX_POSITION_RADIUS;
        }

        SpiralIterator spiral;
        auto startTime = std::chrono::high_resolution_clock::now();
        auto lastPrintTime = startTime;

        if (startX != 0 || startZ != 0) {
            printf("Resuming from position (%d, %d)...\n", startX, startZ);
            while (unlimited || spiral.radius() <= maxRadius) {
                if (spiral.x == startX && spiral.z == startZ) break;
                spiral.next();
            }
            if (spiral.x != startX || spiral.z != startZ) {
                printf("Warning: position (%d, %d) not found in spiral\n", startX, startZ);
                return;
            }
        }

        while (unlimited || spiral.radius() <= maxRadius) {
            // Use optimized k-stride enumeration (K_STRIDE-fold faster than brute force)
            uint64_t found = searchPositionOptimized(spiral.x, spiral.z);

            totalResultsFound += found;
            totalPositionsProcessed++;

            positionLog.seekp(0);
            positionLog << spiral.x << " " << spiral.z << std::endl;
            positionLog.flush();

            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - lastPrintTime).count();

            if (elapsed >= 2.0 || found > 0) {
                lastPrintTime = now;
                double totalElapsed = std::chrono::duration<double>(now - startTime).count();
                double posPerSec = totalPositionsProcessed / totalElapsed;

                printf("\rPos (%d,%d) r=%d | Positions: %lu | Rate: %.1f/s | Found: %lu    ",
                       spiral.x, spiral.z, spiral.radius(),
                       (unsigned long)totalPositionsProcessed, posPerSec,
                       (unsigned long)totalResultsFound);
                fflush(stdout);

                if (found > 0) {
                    printf("\n  -> Found %lu seeds at (%d, %d)!\n",
                           (unsigned long)found, spiral.x, spiral.z);
                }
            }

            spiral.next();
        }

        printf("\n");
    }

    uint64_t getResultCount() const { return totalResultsFound; }
    uint64_t getPositionsProcessed() const { return totalPositionsProcessed; }
};

// ============================================================================
// Host verification
// ============================================================================

bool hostIsSlime(int64_t seed, int32_t x, int32_t z) {
    int64_t posTerm = computePositionTerm(x, z);
    int64_t slimeSeed = seed + posTerm;
    uint64_t internal = ((uint64_t)slimeSeed ^ COMBINED_XOR) & MASK_48;
    uint64_t advanced = (internal * LCG_MULT + LCG_ADD) & MASK_48;
    return (advanced >> 17) % 10 == 0;
}

bool hostCheckPattern(int64_t seed, int32_t baseX, int32_t baseZ) {
    for (int dz = 0; dz < PATTERN_SIZE; dz++) {
        for (int dx = 0; dx < PATTERN_SIZE; dx++) {
            if (!hostIsSlime(seed, baseX + dx, baseZ + dz)) {
                return false;
            }
        }
    }
    return true;
}

// ============================================================================
// Test
// ============================================================================

int runTest() {
    printf("=== %dx%d Slime Pattern Finder Test (Optimized Hensel) ===\n\n",
           PATTERN_SIZE, PATTERN_SIZE);

    initHenselTables();

    printf("Optimization parameters:\n");
    printf("  ROOT_BITS: %d (lower bits solved via Hensel)\n", ROOT_BITS);
    printf("  UPPER_BITS: %d (enumerated after roots found)\n", UPPER_BITS);
    printf("  Speedup: 2^48 -> 2^%d = %.0fx reduction\n", UPPER_BITS, pow(2, 48 - UPPER_BITS));

    printf("\nHensel achievability table:\n");
    for (int r = 1; r <= 8; r++) {
        printf("  %2d output bits: residues {", r);
        bool first = true;
        for (int i = 0; i < 10; i++) {
            if (h_achievableMask[r] & (1 << i)) {
                if (!first) printf(",");
                printf("%d", i);
                first = false;
            }
        }
        printf("}\n");
    }

    // Test cases: (x, z, seed, should_be_valid)
    // Hand-verified test cases
    struct TestCase { int32_t x, z; uint64_t seed; bool expected_valid; };
    TestCase tests[] = {
#if PATTERN_SIZE == 4
        {517, -112, 108601225683165ULL, false},  // Erroneous - not a real 4x4
        {169, 517, 3915317495438ULL, false},     // Erroneous - not a real 4x4
        {64, 333, 98429062619012ULL, true},      // Real - verified 4x4
#elif PATTERN_SIZE == 5
        {51069, 419124, 61692151016558ULL, true},  // Real - verified 5x5
#else
        // No test cases for this pattern size
#endif
    };

    for (const auto& tc : tests) {
        printf("\nTesting seed %llu at (%d, %d)...\n",
               (unsigned long long)tc.seed, tc.x, tc.z);

        bool actual_valid = hostCheckPattern(tc.seed, tc.x, tc.z);

        bool ok = (actual_valid == tc.expected_valid);
        printf("  hostCheckPattern says: %s (expected: %s) %s\n",
               actual_valid ? "VALID" : "INVALID",
               tc.expected_valid ? "VALID" : "INVALID",
               ok ? "OK" : "MISMATCH!");
    }

    // ========================================
    // Test: Search from ~3 rings before target position
    // ========================================
#if PATTERN_SIZE == 4
    int32_t targetX = 64, targetZ = 333;
    uint64_t knownSeed = 98429062619012ULL;
#elif PATTERN_SIZE == 5
    int32_t targetX = 51069, targetZ = 419124;
    uint64_t knownSeed = 61692151016558ULL;
#else
    // Skip search test for unknown pattern sizes
    printf("\nNo search test configured for %dx%d pattern.\n", PATTERN_SIZE, PATTERN_SIZE);
    printf("\nTest complete.\n");
    return 0;
#endif

    printf("\n--- Test: Search starting ~3 rings before (%d, %d) ---\n", targetX, targetZ);

    int32_t offsetX = targetX - 3, offsetZ = targetZ - 3;

    int32_t relTargetX = targetX - offsetX;
    int32_t relTargetZ = targetZ - offsetZ;
    int32_t expectedRadius = std::max(std::abs(relTargetX), std::abs(relTargetZ));

    printf("Target position: (%d, %d)\n", targetX, targetZ);
    printf("Search origin: (%d, %d)\n", offsetX, offsetZ);
    printf("Target relative to origin: (%d, %d) at radius %d\n", relTargetX, relTargetZ, expectedRadius);

    SpiralIterator spiral;
    int32_t searchStartX = offsetX + spiral.x;
    int32_t searchStartZ = offsetZ + spiral.z;
    printf("Search starting position: (%d, %d)\n", searchStartX, searchStartZ);

    OptimizedHenselSlimeSearcher searcher;

    auto startTime = std::chrono::high_resolution_clock::now();
    uint64_t positionsSearched = 0;
    uint64_t totalSeedsFound = 0;
    bool foundTarget = false;
    uint64_t targetSeeds = 0;

    while (spiral.radius() <= expectedRadius + 1) {
        int32_t currentX = offsetX + spiral.x;
        int32_t currentZ = offsetZ + spiral.z;

        printf("\r  Searching (%d, %d) ring %d, pos %lu...    ",
               currentX, currentZ, spiral.radius(), (unsigned long)positionsSearched);
        fflush(stdout);

        uint64_t found = searcher.searchPosition(currentX, currentZ);
        positionsSearched++;
        totalSeedsFound += found;

        if (currentX == targetX && currentZ == targetZ) {
            foundTarget = true;
            targetSeeds = found;
            printf("\n  -> Reached target (%d, %d) at ring %d! Found %lu seeds.\n",
                   targetX, targetZ, spiral.radius(), (unsigned long)found);
            break;
        }

        spiral.next();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(endTime - startTime).count();

    printf("\nSearch results:\n");
    printf("  Positions searched: %lu\n", (unsigned long)positionsSearched);
    printf("  Total seeds found: %lu\n", (unsigned long)totalSeedsFound);
    printf("  Time elapsed: %.2f seconds\n", elapsed);
    printf("  Rate: %.1f positions/second\n", positionsSearched / elapsed);

    if (foundTarget) {
        printf("  Target (%d, %d) found: YES with %lu seeds\n", targetX, targetZ, (unsigned long)targetSeeds);

        bool knownValid = hostCheckPattern(knownSeed, targetX, targetZ);
        printf("  Known seed %llu verification: %s\n",
               (unsigned long long)knownSeed,
               knownValid ? "VALID" : "INVALID");
    } else {
        printf("  Target (%d, %d) found: NO (unexpected!)\n", targetX, targetZ);
    }

    printf("\nTest complete.\n");
    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    printf("Slime Chunk Finder - %dx%d Pattern (Optimized Hensel)\n", PATTERN_SIZE, PATTERN_SIZE);
    printf("======================================================\n\n");

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test") == 0) return runTest();
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--test] [--radius N] [--start X Z]\n", argv[0]);
            printf("\nOptimized Hensel + K-Stride: finds lower-bit roots, then uses k-stride optimization.\n");
            printf("K-stride optimization: valid k values satisfy k ≡ offset (mod %llu)\n", (unsigned long long)K_STRIDE);
            printf("This reduces enumeration from 2^%d to 2^%d per root (~%llux speedup).\n",
                   UPPER_BITS, UPPER_BITS - K_STRIDE_BITS, (unsigned long long)K_STRIDE);
            printf("\nOptions:\n");
            printf("  --test      Run tests\n");
            printf("  --radius N  Search up to radius N (0 or omit for unlimited)\n");
            printf("  --start X Z Resume from position (X, Z)\n");
            printf("\nOutput: slime_%dx%d_results.txt\n", PATTERN_SIZE, PATTERN_SIZE);
            printf("Position log: slime_current_position.txt\n");
            return 0;
        }
    }

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices!\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%.1f GB, %d SMs)\n", prop.name,
           prop.totalGlobalMem / 1e9, prop.multiProcessorCount);

    int32_t searchRadius = 0;
    int32_t startX = 0, startZ = 0;
    bool manualStart = false;

    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--radius") == 0) {
            searchRadius = atoi(argv[i + 1]);
        }
        if (strcmp(argv[i], "--start") == 0 && i + 2 < argc) {
            startX = atoi(argv[i + 1]);
            startZ = atoi(argv[i + 2]);
            manualStart = true;
        }
    }

    if (!manualStart) {
        std::ifstream posFile("slime_current_position.txt");
        if (posFile.is_open()) {
            if (posFile >> startX >> startZ) {
                printf("Auto-resuming from slime_current_position.txt: (%d, %d)\n", startX, startZ);
            }
            posFile.close();
        }
    }

    printf("Pattern: %dx%d (%d constraints)\n", PATTERN_SIZE, PATTERN_SIZE, NUM_CONSTRAINTS);
    printf("Method: Hensel + K-Stride (roots at %d bits, k-stride=%llu, ~%llux speedup)\n",
           ROOT_BITS, (unsigned long long)K_STRIDE, (unsigned long long)K_STRIDE);
    if (searchRadius <= 0) {
        printf("Search radius: unlimited\n");
    } else {
        printf("Search radius: %d positions from origin\n", searchRadius);
    }
    if (startX != 0 || startZ != 0) {
        printf("Starting position: (%d, %d)\n", startX, startZ);
    }
    printf("\n");

    OptimizedHenselSlimeSearcher searcher;
    searcher.search(searchRadius, startX, startZ);

    printf("\nComplete. Positions: %lu, Results: %lu\n",
           (unsigned long)searcher.getPositionsProcessed(),
           (unsigned long)searcher.getResultCount());

    return 0;
}
