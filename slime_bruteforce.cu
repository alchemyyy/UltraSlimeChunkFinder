/**
 * Slime Chunk 4x4 Brute Force Finder
 * 
 * Brute forces all 2^48 seeds to find which produce a 4x4 slime chunk
 * pattern at a user-specified position.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>

// ============================================================================
// Constants
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

// CUDA parameters
constexpr int THREADS_PER_BLOCK = 256;
constexpr uint64_t BATCH_SIZE = 1ULL << 30;  // 1 billion seeds per batch
constexpr uint32_t MAX_RESULTS = 1U << 24;   // 16M max results per batch

// ============================================================================
// Device: Slime chunk check
// ============================================================================

__device__ __forceinline__
int64_t computePositionTerm(int32_t x, int32_t z) {
    int32_t term1 = x * x * SLIME_A;
    int32_t term2 = x * SLIME_B;
    int64_t term3 = (int64_t)(z * z) * SLIME_C;
    int32_t term4 = z * SLIME_D;
    return (int64_t)term1 + term2 + term3 + term4;
}

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
// Kernel: Brute force seeds
// ============================================================================

__global__ void bruteForceKernel(
    uint64_t seedStart,
    uint64_t seedCount,
    int32_t baseX,
    int32_t baseZ,
    uint64_t* results,
    uint32_t* resultCount,
    uint32_t maxResults
) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= seedCount) return;

    uint64_t seed = seedStart + idx;
    
    if (checkPattern(seed, baseX, baseZ)) {
        uint32_t pos = atomicAdd(resultCount, 1);
        if (pos < maxResults) {
            results[pos] = seed;
        }
    }
}

// ============================================================================
// Host verification
// ============================================================================

bool hostIsSlime(uint64_t seed, int32_t x, int32_t z) {
    int32_t term1 = x * x * SLIME_A;
    int32_t term2 = x * SLIME_B;
    int64_t term3 = (int64_t)(z * z) * SLIME_C;
    int32_t term4 = z * SLIME_D;
    int64_t posTerm = (int64_t)term1 + term2 + term3 + term4;
    
    int64_t slimeSeed = (int64_t)seed + posTerm;
    uint64_t internal = ((uint64_t)slimeSeed ^ COMBINED_XOR) & MASK_48;
    uint64_t advanced = (internal * LCG_MULT + LCG_ADD) & MASK_48;
    return (advanced >> 17) % 10 == 0;
}

bool hostCheckPattern(uint64_t seed, int32_t baseX, int32_t baseZ) {
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
// Main
// ============================================================================

int main(int argc, char** argv) {
    printf("=== 4x4 Slime Chunk Brute Force Finder ===\n\n");

    if (argc < 3) {
        printf("Usage: %s <baseX> <baseZ>\n", argv[0]);
        printf("\nBrute forces all 2^48 seeds to find 4x4 slime patterns.\n");
        printf("Results are written to slime_bruteforce_results.txt\n");
        return 1;
    }

    int32_t baseX = atoi(argv[1]);
    int32_t baseZ = atoi(argv[2]);

    printf("Position: (%d, %d)\n", baseX, baseZ);
    printf("Pattern: %dx%d slime chunks\n", PATTERN_SIZE, PATTERN_SIZE);
    printf("Search space: 2^48 = 281,474,976,710,656 seeds\n\n");

    // Check CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("ERROR: No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%.1f GB, %d SMs)\n\n", prop.name,
           prop.totalGlobalMem / 1e9, prop.multiProcessorCount);

    // Allocate GPU memory
    uint64_t* d_results;
    uint32_t* d_resultCount;
    cudaMalloc(&d_results, MAX_RESULTS * sizeof(uint64_t));
    cudaMalloc(&d_resultCount, sizeof(uint32_t));

    // Open output file
    char filename[256];
    snprintf(filename, sizeof(filename), "slime_bruteforce_results_%d_%d.txt", baseX, baseZ);
    FILE* outFile = fopen(filename, "w");
    if (!outFile) {
        printf("ERROR: Cannot open output file!\n");
        return 1;
    }
    fprintf(outFile, "# 4x4 Slime Chunk Seeds for position (%d, %d)\n", baseX, baseZ);
    fprintf(outFile, "# Format: seed\n");

    // Host buffer for results
    std::vector<uint64_t> h_results(MAX_RESULTS);

    uint64_t totalSeeds = 1ULL << 48;
    uint64_t totalFound = 0;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    auto lastPrintTime = startTime;

    printf("Starting brute force search...\n\n");

    for (uint64_t batchStart = 0; batchStart < totalSeeds; batchStart += BATCH_SIZE) {
        uint64_t batchCount = std::min(BATCH_SIZE, totalSeeds - batchStart);

        // Reset result count
        cudaMemset(d_resultCount, 0, sizeof(uint32_t));

        // Launch kernel
        uint64_t blocks = (batchCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        bruteForceKernel<<<blocks, THREADS_PER_BLOCK>>>(
            batchStart, batchCount, baseX, baseZ,
            d_results, d_resultCount, MAX_RESULTS
        );
        cudaDeviceSynchronize();

        // Get results
        uint32_t numFound;
        cudaMemcpy(&numFound, d_resultCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        if (numFound > 0) {
            numFound = std::min(numFound, MAX_RESULTS);
            cudaMemcpy(h_results.data(), d_results, numFound * sizeof(uint64_t), cudaMemcpyDeviceToHost);

            // Verify and write results
            for (uint32_t i = 0; i < numFound; i++) {
                uint64_t seed = h_results[i];
                if (hostCheckPattern(seed, baseX, baseZ)) {
                    fprintf(outFile, "%llu\n", (unsigned long long)seed);
                    totalFound++;
                }
            }
            fflush(outFile);
        }

        // Progress update
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - lastPrintTime).count();

        if (elapsed >= 2.0 || batchStart + batchCount >= totalSeeds) {
            lastPrintTime = now;
            double totalElapsed = std::chrono::duration<double>(now - startTime).count();
            double progress = (double)(batchStart + batchCount) / totalSeeds * 100.0;
            double seedsPerSec = (batchStart + batchCount) / totalElapsed;
            double eta = (totalSeeds - batchStart - batchCount) / seedsPerSec;

            printf("\rProgress: %.2f%% | Seeds: %.2fT / %.2fT | Rate: %.2fB/s | Found: %llu | ETA: %.0fs    ",
                   progress,
                   (batchStart + batchCount) / 1e12,
                   totalSeeds / 1e12,
                   seedsPerSec / 1e9,
                   (unsigned long long)totalFound,
                   eta);
            fflush(stdout);
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(endTime - startTime).count();

    printf("\n\n=== COMPLETE ===\n");
    printf("Total seeds checked: %llu\n", (unsigned long long)totalSeeds);
    printf("Total seeds found: %llu\n", (unsigned long long)totalFound);
    printf("Total time: %.2f seconds\n", totalTime);
    printf("Average rate: %.2f billion seeds/second\n", totalSeeds / totalTime / 1e9);
    printf("\nResults written to: %s\n", filename);

    fclose(outFile);
    cudaFree(d_results);
    cudaFree(d_resultCount);

    return 0;
}
