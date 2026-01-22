/**
 * True Hensel Lifting Slime Chunk Finder with Stride Optimization
 *
 * APPROACH:
 * 1. Iterate positions in spiral order from (0,0)
 * 2. Use Hensel lifting to find valid roots (lower START_BIT bits)
 * 3. STRIDE OPTIMIZATION:
 *    - For each root, enumerate upper bits sequentially
 *    - Find stride by discovering 2+ seeds from the same root
 *    - Refine stride using GCD and divisor testing
 *    - Generate all seeds using arithmetic progression
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <numeric>

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

// Pattern size
constexpr int PATTERN_SIZE = 4;
constexpr int NUM_CONSTRAINTS = PATTERN_SIZE * PATTERN_SIZE;

// Hensel parameters
constexpr int START_BIT = 18;
constexpr int END_BIT = 48;
constexpr uint32_t MAX_CANDIDATES = 1U << 26;  // 64M max candidates
constexpr uint32_t MAX_RESULTS = 1U << 20;
constexpr int32_t MAX_POSITION_RADIUS = 1875000;

// Stride discovery parameters
constexpr uint64_t UPPER_BITS = END_BIT - START_BIT;  // 30 upper bits
constexpr uint64_t TOTAL_UPPER = 1ULL << UPPER_BITS;   // 2^30 combinations
constexpr uint64_t BATCH_SIZE = 1ULL << 24;            // 16M per batch

// CUDA launch parameters
// 256 threads per block is a common choice because:
// - It's a multiple of warp size (32), ensuring full warp utilization
// - Provides good occupancy on most GPUs (typically 1024-2048 max threads/SM)
// - Balances register usage vs parallelism
// - 256 = 8 warps, which hides memory latency well
// - Works well for compute-bound kernels like ours
constexpr int THREADS_PER_BLOCK = 256;

// ============================================================================
// Hensel Lookup Table
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
// Host/Device: Position term computation
// ============================================================================

__host__ __device__ __forceinline__
int64_t computePositionTerm(int32_t x, int32_t z) {
    int32_t term1 = x * x * SLIME_A;
    int32_t term2 = x * SLIME_B;
    int64_t term3 = (int64_t)(z * z) * SLIME_C;
    int32_t term4 = z * SLIME_D;
    return (int64_t)term1 + term2 + term3 + term4;
}

// ============================================================================
// Device: Full verification
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
// Device: Hensel constraint check
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
// Kernel: Initialize candidates at START_BIT
// ============================================================================

__global__ void initCandidates(
    const int64_t* __restrict__ posTerms,
    uint64_t* candidates,
    uint32_t* candidateCount,
    uint32_t maxCandidates
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t totalCandidates = 1U << START_BIT;

    if (idx >= totalCandidates) return;

    uint64_t candidate = idx;

    if (canSatisfyAllConstraints(candidate, START_BIT, posTerms)) {
        uint32_t pos = atomicAdd(candidateCount, 1);
        if (pos < maxCandidates) {
            candidates[pos] = candidate;
        }
    }
}

// ============================================================================
// Kernel: Expand and filter at next bit level
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
// Kernel: Enumerate upper bits for a single root
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

    uint64_t upper = upperStart + idx;
    uint64_t seed = root | (upper << START_BIT);

    if (checkPattern(seed, baseX, baseZ)) {
        uint32_t pos = atomicAdd(resultCount, 1);
        if (pos < maxResults) {
            results[pos] = seed;
        }
    }
}

// ============================================================================
// Kernel: Generate seeds with stride
// ============================================================================

__global__ void generateSeedsWithStride(
    const uint64_t* __restrict__ baseSeeds,
    uint32_t numBaseSeeds,
    uint64_t stride,
    int32_t baseX, int32_t baseZ,
    uint64_t* results,
    uint32_t* resultCount,
    uint32_t maxResults
) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    
    // Calculate max multiples possible
    uint64_t maxMultiple = MASK_48 / stride;
    uint64_t totalWork = (uint64_t)numBaseSeeds * (maxMultiple + 1);
    
    if (idx >= totalWork) return;
    
    uint32_t baseIdx = idx / (maxMultiple + 1);
    uint64_t multiple = idx % (maxMultiple + 1);
    
    if (baseIdx >= numBaseSeeds) return;
    
    uint64_t seed = baseSeeds[baseIdx] + multiple * stride;
    if (seed > MASK_48) return;

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
// Host verification functions
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
// GCD helper
// ============================================================================

uint64_t gcd(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

// ============================================================================
// Get divisors of a number
// ============================================================================

std::vector<uint64_t> getDivisors(uint64_t n) {
    std::vector<uint64_t> divisors;
    for (uint64_t i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            divisors.push_back(i);
            if (i != n / i) {
                divisors.push_back(n / i);
            }
        }
    }
    std::sort(divisors.begin(), divisors.end());
    return divisors;
}

// ============================================================================
// Stride-based Slime Searcher
// ============================================================================

class StrideSlimeSearcher {
private:
    // GPU resources
    int64_t* d_posTerms;
    uint64_t* d_candidates[2];
    uint32_t* d_candidateCount;
    uint64_t* d_results;
    uint32_t* d_resultCount;
    uint64_t* d_baseSeeds;

    // Host resources
    std::vector<int64_t> h_posTerms;
    std::ofstream outFile;
    std::ofstream positionLog;

    uint64_t totalPositionsProcessed;
    uint64_t totalResultsFound;

public:
    StrideSlimeSearcher() : totalPositionsProcessed(0), totalResultsFound(0) {
        initHenselTables();

        h_posTerms.resize(NUM_CONSTRAINTS);

        cudaMalloc(&d_posTerms, NUM_CONSTRAINTS * sizeof(int64_t));
        cudaMalloc(&d_candidates[0], MAX_CANDIDATES * sizeof(uint64_t));
        cudaMalloc(&d_candidates[1], MAX_CANDIDATES * sizeof(uint64_t));
        cudaMalloc(&d_candidateCount, sizeof(uint32_t));
        cudaMalloc(&d_results, MAX_RESULTS * sizeof(uint64_t));
        cudaMalloc(&d_resultCount, sizeof(uint32_t));
        cudaMalloc(&d_baseSeeds, MAX_RESULTS * sizeof(uint64_t));

        char filename[64];
        snprintf(filename, sizeof(filename), "slime_%dx%d_results.txt", PATTERN_SIZE, PATTERN_SIZE);
        outFile.open(filename, std::ios::app);
        outFile << "# " << PATTERN_SIZE << "x" << PATTERN_SIZE << " Slime Pattern Results (Stride)\n";
        outFile << "# Format: baseX,baseZ,seed\n";
        outFile.flush();

        positionLog.open("slime_current_position.txt", std::ios::out);
    }

    ~StrideSlimeSearcher() {
        outFile.close();
        positionLog.close();
        cudaFree(d_posTerms);
        cudaFree(d_candidates[0]);
        cudaFree(d_candidates[1]);
        cudaFree(d_candidateCount);
        cudaFree(d_results);
        cudaFree(d_resultCount);
        cudaFree(d_baseSeeds);
    }

    void computePositionTerms(int32_t baseX, int32_t baseZ) {
        for (int dz = 0; dz < PATTERN_SIZE; dz++) {
            for (int dx = 0; dx < PATTERN_SIZE; dx++) {
                h_posTerms[dz * PATTERN_SIZE + dx] = computePositionTerm(baseX + dx, baseZ + dz);
            }
        }
        cudaMemcpy(d_posTerms, h_posTerms.data(), NUM_CONSTRAINTS * sizeof(int64_t), cudaMemcpyHostToDevice);
    }

    // Phase 1: Find Hensel roots (lower START_BIT bits)
    std::vector<uint64_t> findRoots() {
        int currentBuffer = 0;

        // Initialize at START_BIT
        cudaMemset(d_candidateCount, 0, sizeof(uint32_t));
        uint32_t initCount = 1U << START_BIT;
        int initBlocks = (initCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        initCandidates<<<initBlocks, THREADS_PER_BLOCK>>>(
            d_posTerms,
            d_candidates[currentBuffer],
            d_candidateCount,
            MAX_CANDIDATES
        );
        cudaDeviceSynchronize();

        uint32_t numRoots;
        cudaMemcpy(&numRoots, d_candidateCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        if (numRoots == 0) return {};

        numRoots = std::min(numRoots, MAX_CANDIDATES);
        std::vector<uint64_t> roots(numRoots);
        cudaMemcpy(roots.data(), d_candidates[currentBuffer], numRoots * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        return roots;
    }

    // Enumerate upper bits for a root to find seeds
    std::vector<uint64_t> enumerateRootSeeds(uint64_t root, int32_t baseX, int32_t baseZ, 
                                              uint64_t upperStart, uint64_t upperCount) {
        cudaMemset(d_resultCount, 0, sizeof(uint32_t));

        uint64_t blocks = (upperCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        blocks = std::min(blocks, (uint64_t)(65535 * 256));

        enumerateSingleRoot<<<blocks, THREADS_PER_BLOCK>>>(
            root, baseX, baseZ,
            d_results, d_resultCount, MAX_RESULTS,
            upperStart, upperCount
        );
        cudaDeviceSynchronize();

        uint32_t numSeeds;
        cudaMemcpy(&numSeeds, d_resultCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        if (numSeeds == 0) return {};

        numSeeds = std::min(numSeeds, MAX_RESULTS);
        std::vector<uint64_t> seeds(numSeeds);
        cudaMemcpy(seeds.data(), d_results, numSeeds * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        return seeds;
    }

    // Find stride by discovering 2+ seeds from any root
    uint64_t discoverStride(const std::vector<uint64_t>& roots, int32_t baseX, int32_t baseZ,
                            std::map<uint64_t, std::vector<uint64_t>>& seedsByRoot) {
        
        // Enumerate in batches until we find 2 seeds from the same root
        for (uint64_t batchStart = 0; batchStart < TOTAL_UPPER; batchStart += BATCH_SIZE) {
            uint64_t batchCount = std::min(BATCH_SIZE, TOTAL_UPPER - batchStart);

            for (uint64_t root : roots) {
                auto newSeeds = enumerateRootSeeds(root, baseX, baseZ, batchStart, batchCount);
                
                for (uint64_t seed : newSeeds) {
                    seedsByRoot[root].push_back(seed);
                }

                // Check if we have 2+ seeds from this root
                if (seedsByRoot[root].size() >= 2) {
                    std::sort(seedsByRoot[root].begin(), seedsByRoot[root].end());
                    uint64_t stride = seedsByRoot[root][1] - seedsByRoot[root][0];
                    return stride;
                }
            }
        }

        return 0;  // No stride found
    }

    // Refine stride using GCD and divisor testing
    uint64_t refineStride(uint64_t initialStride, const std::map<uint64_t, std::vector<uint64_t>>& seedsByRoot,
                          int32_t baseX, int32_t baseZ) {
        if (initialStride == 0) return 0;

        // Compute GCD of all differences
        uint64_t strideGcd = initialStride;
        for (auto& [root, seeds] : seedsByRoot) {
            if (seeds.size() < 2) continue;
            for (size_t i = 1; i < seeds.size(); i++) {
                uint64_t diff = seeds[i] - seeds[i-1];
                strideGcd = gcd(strideGcd, diff);
            }
        }

        // Test divisors to find true stride
        std::vector<uint64_t> divisors = getDivisors(strideGcd);
        
        // Get a base seed for testing
        uint64_t baseSeed = 0;
        for (auto& [root, seeds] : seedsByRoot) {
            if (!seeds.empty()) {
                baseSeed = seeds[0];
                break;
            }
        }

        for (uint64_t d : divisors) {
            if (d == 0) continue;

            bool plusValid = (baseSeed + d <= MASK_48) && hostCheckPattern(baseSeed + d, baseX, baseZ);
            bool minusValid = (baseSeed >= d) && hostCheckPattern(baseSeed - d, baseX, baseZ);

            if (plusValid || minusValid) {
                return d;
            }
        }

        return strideGcd;
    }

    // Get base seeds (one per congruence class)
    std::vector<uint64_t> getBaseSeeds(const std::map<uint64_t, std::vector<uint64_t>>& seedsByRoot, 
                                        uint64_t stride) {
        std::map<uint64_t, uint64_t> classToSeed;

        for (auto& [root, seeds] : seedsByRoot) {
            for (uint64_t seed : seeds) {
                uint64_t classId = seed % stride;
                if (classToSeed.find(classId) == classToSeed.end() || seed < classToSeed[classId]) {
                    classToSeed[classId] = seed;
                }
            }
        }

        // Find minimum seed in each class
        std::vector<uint64_t> baseSeeds;
        for (auto& [classId, seed] : classToSeed) {
            uint64_t minSeed = seed % stride;
            if (minSeed == 0) minSeed = classId;  // classId is already seed % stride
            baseSeeds.push_back(minSeed > 0 ? minSeed : stride);
        }

        // Remove duplicates and sort
        std::sort(baseSeeds.begin(), baseSeeds.end());
        baseSeeds.erase(std::unique(baseSeeds.begin(), baseSeeds.end()), baseSeeds.end());

        return baseSeeds;
    }

    uint64_t searchPosition(int32_t baseX, int32_t baseZ) {
        computePositionTerms(baseX, baseZ);

        // Phase 1: Find Hensel roots
        std::vector<uint64_t> roots = findRoots();
        if (roots.empty()) return 0;

        // Phase 2: Discover stride by enumerating upper bits
        std::map<uint64_t, std::vector<uint64_t>> seedsByRoot;
        uint64_t stride = discoverStride(roots, baseX, baseZ, seedsByRoot);

        if (stride == 0) {
            // No seeds found at all
            return 0;
        }

        // Phase 3: Refine stride
        stride = refineStride(stride, seedsByRoot, baseX, baseZ);

        // Phase 4: Continue enumeration to find all congruence classes
        // (enumerate at least one full stride period)
        uint64_t periodBatches = (stride / BATCH_SIZE) + 2;
        uint64_t totalBatches = std::min(periodBatches, TOTAL_UPPER / BATCH_SIZE);

        for (uint64_t batch = 0; batch < totalBatches; batch++) {
            uint64_t batchStart = batch * BATCH_SIZE;
            uint64_t batchCount = std::min(BATCH_SIZE, TOTAL_UPPER - batchStart);

            for (uint64_t root : roots) {
                auto newSeeds = enumerateRootSeeds(root, baseX, baseZ, batchStart, batchCount);
                for (uint64_t seed : newSeeds) {
                    if (std::find(seedsByRoot[root].begin(), seedsByRoot[root].end(), seed) == seedsByRoot[root].end()) {
                        seedsByRoot[root].push_back(seed);
                    }
                }
            }
        }

        // Phase 5: Get base seeds and generate all results
        std::vector<uint64_t> baseSeeds = getBaseSeeds(seedsByRoot, stride);
        
        uint64_t totalResults = 0;

        // Generate all seeds using stride
        for (uint64_t baseSeed : baseSeeds) {
            for (uint64_t seed = baseSeed; seed <= MASK_48; seed += stride) {
                if (hostCheckPattern(seed, baseX, baseZ)) {
                    outFile << baseX << "," << baseZ << "," << seed << "\n";
                    totalResults++;
                }
            }
        }
        outFile.flush();

        return totalResults;
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
            uint64_t found = searchPosition(spiral.x, spiral.z);

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
// Test
// ============================================================================

int runTest() {
    printf("=== %dx%d Slime Pattern Finder Test (Stride) ===\n\n", PATTERN_SIZE, PATTERN_SIZE);

    initHenselTables();

    printf("Parameters:\n");
    printf("  START_BIT: %d (2^%d = %u roots)\n", START_BIT, START_BIT, 1U << START_BIT);
    printf("  END_BIT: %d\n", END_BIT);
    printf("  UPPER_BITS: %llu (2^%llu combinations)\n", (unsigned long long)UPPER_BITS, (unsigned long long)UPPER_BITS);
    printf("  BATCH_SIZE: %llu\n", (unsigned long long)BATCH_SIZE);

    // Test 1: Verify known pattern
    printf("\n--- Test 1: Verify known result ---\n");
    int32_t knownX = 64, knownZ = 333;
    uint64_t knownSeed = 100147049537412ULL;

    printf("Known: position (%d, %d), seed %llu\n", knownX, knownZ, (unsigned long long)knownSeed);

    bool allSlime = true;
    printf("  Pattern:\n");
    for (int dz = 0; dz < PATTERN_SIZE; dz++) {
        printf("  ");
        for (int dx = 0; dx < PATTERN_SIZE; dx++) {
            bool slime = hostIsSlime(knownSeed, knownX + dx, knownZ + dz);
            printf("%s ", slime ? "##" : "..");
            if (!slime) allSlime = false;
        }
        printf("\n");
    }

    if (allSlime) {
        printf("  Result: VALID %dx%d pattern!\n", PATTERN_SIZE, PATTERN_SIZE);
    } else {
        printf("  Result: FAILED - not all chunks are slime!\n");
        return 1;
    }

    // Test 2: Search at known position
    printf("\n--- Test 2: Search at (64, 333) ---\n");

    StrideSlimeSearcher searcher;

    auto start = std::chrono::high_resolution_clock::now();
    uint64_t found = searcher.searchPosition(knownX, knownZ);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    printf("Found %lu seeds at (%d, %d) in %.3f seconds\n", (unsigned long)found, knownX, knownZ, elapsed);

    if (found > 0) {
        printf("  SUCCESS: Seeds found!\n");
    } else {
        printf("  WARNING: No seeds found (expected to find seed %llu)\n", (unsigned long long)knownSeed);
    }

    // Test 3: Search at origin
    printf("\n--- Test 3: Search at (0, 0) ---\n");

    auto start2 = std::chrono::high_resolution_clock::now();
    uint64_t found2 = searcher.searchPosition(0, 0);
    auto end2 = std::chrono::high_resolution_clock::now();
    double elapsed2 = std::chrono::duration<double>(end2 - start2).count();

    printf("Found %lu seeds at (0, 0) in %.3f seconds\n", (unsigned long)found2, elapsed2);

    printf("\nAll tests complete.\n");
    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    printf("Slime Chunk Finder - %dx%d Pattern (Stride Optimization)\n", PATTERN_SIZE, PATTERN_SIZE);
    printf("=========================================================\n\n");

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test") == 0) return runTest();
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--test] [--radius N] [--start X Z]\n", argv[0]);
            printf("\nHensel root finding + stride-based seed generation.\n");
            printf("\nOptions:\n");
            printf("  --test      Run tests\n");
            printf("  --radius N  Search up to radius N\n");
            printf("  --start X Z Resume from position\n");
            printf("\nOutput: slime_%dx%d_results.txt\n", PATTERN_SIZE, PATTERN_SIZE);
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
                printf("Auto-resuming: (%d, %d)\n", startX, startZ);
            }
            posFile.close();
        }
    }

    printf("Pattern: %dx%d (%d constraints)\n", PATTERN_SIZE, PATTERN_SIZE, NUM_CONSTRAINTS);
    printf("Method: Hensel roots + Stride enumeration\n");
    printf("\n");

    StrideSlimeSearcher searcher;
    searcher.search(searchRadius, startX, startZ);

    printf("\nComplete. Positions: %lu, Results: %lu\n",
           (unsigned long)searcher.getPositionsProcessed(),
           (unsigned long)searcher.getResultCount());

    return 0;
}
