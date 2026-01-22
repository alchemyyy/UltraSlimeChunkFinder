/**
 * Ultra Slime Chunk Finder - User-Defined Pattern Search
 *
 * Loads pattern from pattern.txt:
 *   # = must be slime
 *   . = must NOT be slime
 *   X = don't care
 *
 * Optimized for RTX 4080 Super (12GB VRAM) + Threadripper 9960X
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>

// ============================================================================
// Constants
// ============================================================================

constexpr uint64_t MASK_48 = (1ULL << 48) - 1;
constexpr uint64_t LCG_MULT = 0x5DEECE66DULL;
constexpr uint64_t LCG_ADD = 0xBULL;
constexpr uint64_t LCG_MULT_INV = 0xDFE05BCB1365ULL;
constexpr uint64_t XOR_CONST = 0x3ad8025fULL;

// Slime chunk formula constants
constexpr int32_t SLIME_A = 0x4c1906;
constexpr int32_t SLIME_B = 0x5ac0db;
constexpr int64_t SLIME_C = 0x4307a7LL;
constexpr int32_t SLIME_D = 0x5f24f;

// Tile and batch sizes
constexpr int32_t TILE_SIZE = 4096;
constexpr uint64_t SEED_BATCH_SIZE = 1ULL << 20;  // 1M seeds per batch
constexpr uint32_t MAX_MATCHES_PER_BATCH = 1 << 20;
constexpr int NUM_CUDA_STREAMS = 8;
constexpr int NUM_FFT_SLOTS = 16;

// ============================================================================
// Pattern Definition - loaded from file
// ============================================================================

struct Pattern {
    std::vector<std::vector<int8_t>> grid;  // 1 = must slime, -1 = must not slime, 0 = don't care
    int32_t width;
    int32_t height;
    int32_t mustSlimeCount;      // Number of # cells
    int32_t mustNotSlimeCount;   // Number of . cells
    bool hasAntiPattern;         // True if any . cells exist
    
    Pattern() : width(0), height(0), mustSlimeCount(0), mustNotSlimeCount(0), hasAntiPattern(false) {}
    
    bool load(const char* filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            printf("Error: Cannot open pattern file '%s'\n", filename);
            return false;
        }
        
        grid.clear();
        std::string line;
        width = 0;
        
        while (std::getline(file, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '/' || line[0] == ';') continue;
            
            std::vector<int8_t> row;
            for (char c : line) {
                if (c == '#') {
                    row.push_back(1);
                    mustSlimeCount++;
                } else if (c == '.') {
                    row.push_back(-1);
                    mustNotSlimeCount++;
                } else if (c == 'X' || c == 'x' || c == ' ' || c == '?') {
                    row.push_back(0);
                }
                // Ignore other characters (like \r)
            }
            
            if (!row.empty()) {
                width = std::max(width, (int32_t)row.size());
                grid.push_back(row);
            }
        }
        
        height = (int32_t)grid.size();
        hasAntiPattern = (mustNotSlimeCount > 0);
        
        // Pad rows to uniform width
        for (auto& row : grid) {
            while ((int32_t)row.size() < width) {
                row.push_back(0);  // Pad with "don't care"
            }
        }
        
        return height > 0 && width > 0;
    }
    
    void print() const {
        printf("Pattern (%dx%d):\n", width, height);
        for (int z = 0; z < height; z++) {
            printf("  ");
            for (int x = 0; x < width; x++) {
                char c = (grid[z][x] == 1) ? '#' : (grid[z][x] == -1) ? '.' : 'X';
                printf("%c", c);
            }
            printf("\n");
        }
        printf("Must be slime: %d cells\n", mustSlimeCount);
        printf("Must NOT be slime: %d cells\n", mustNotSlimeCount);
        printf("Anti-pattern check: %s\n", hasAntiPattern ? "ENABLED" : "DISABLED (faster)");
    }
};

// Global pattern
Pattern g_pattern;

// ============================================================================
// Spiral Tile Iterator
// ============================================================================

class SpiralIterator {
private:
    int32_t x, z, dx, dz, segmentLength, segmentPassed;

public:
    SpiralIterator() : x(0), z(0), dx(1), dz(0), segmentLength(1), segmentPassed(0) {}
    
    void reset() { x = z = 0; dx = 1; dz = 0; segmentLength = 1; segmentPassed = 0; }
    int32_t tileX() const { return x; }
    int32_t tileZ() const { return z; }
    
    void getChunkBounds(int32_t& minX, int32_t& minZ, int32_t& maxX, int32_t& maxZ) const {
        minX = x * TILE_SIZE - TILE_SIZE / 2;
        minZ = z * TILE_SIZE - TILE_SIZE / 2;
        maxX = minX + TILE_SIZE - 1;
        maxZ = minZ + TILE_SIZE - 1;
    }
    
    int32_t distance() const { return std::max(std::abs(x), std::abs(z)); }
    
    void next() {
        x += dx; z += dz; segmentPassed++;
        if (segmentPassed == segmentLength) {
            segmentPassed = 0;
            int32_t temp = dx; dx = -dz; dz = temp;
            if (dz == 0) segmentLength++;
        }
    }
};

// ============================================================================
// Device: Slime chunk check
// ============================================================================

__device__ __forceinline__ bool isSlimeChunkDevice(int64_t worldSeed, int32_t chunkX, int32_t chunkZ) {
    int64_t term1 = static_cast<int64_t>(chunkX * chunkX * SLIME_A);
    int64_t term2 = static_cast<int64_t>(chunkX * SLIME_B);
    int64_t term3 = static_cast<int64_t>(chunkZ * chunkZ) * SLIME_C;
    int64_t term4 = static_cast<int64_t>(chunkZ * SLIME_D);
    int64_t slimeSeed = (worldSeed + term1 + term2 + term3 + term4) ^ XOR_CONST;
    uint64_t internal = (static_cast<uint64_t>(slimeSeed) ^ LCG_MULT) & MASK_48;
    uint64_t advanced = (internal * LCG_MULT + LCG_ADD) & MASK_48;
    return (advanced >> 17) % 10 == 0;
}

// ============================================================================
// Host: Slime chunk check
// ============================================================================

inline bool isSlimeChunkHost(int64_t worldSeed, int32_t chunkX, int32_t chunkZ) {
    int64_t term1 = static_cast<int64_t>(chunkX * chunkX * SLIME_A);
    int64_t term2 = static_cast<int64_t>(chunkX * SLIME_B);
    int64_t term3 = static_cast<int64_t>(chunkZ * chunkZ) * SLIME_C;
    int64_t term4 = static_cast<int64_t>(chunkZ * SLIME_D);
    int64_t slimeSeed = (worldSeed + term1 + term2 + term3 + term4) ^ XOR_CONST;
    uint64_t internal = (static_cast<uint64_t>(slimeSeed) ^ LCG_MULT) & MASK_48;
    uint64_t advanced = (internal * LCG_MULT + LCG_ADD) & MASK_48;
    return (advanced >> 17) % 10 == 0;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void generateTileBitmapKernel(
    int64_t seed, float* bitmap, int32_t tileSize,
    int32_t tileOffsetX, int32_t tileOffsetZ
) {
    int32_t localX = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t localZ = blockIdx.y * blockDim.y + threadIdx.y;
    if (localX < tileSize && localZ < tileSize) {
        bitmap[localZ * tileSize + localX] = 
            isSlimeChunkDevice(seed, tileOffsetX + localX, tileOffsetZ + localZ) ? 1.0f : 0.0f;
    }
}

// Inverted bitmap for anti-pattern matching (1 where NOT slime)
__global__ void generateInvertedBitmapKernel(
    const float* bitmap, float* inverted, int32_t size
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        inverted[idx] = 1.0f - bitmap[idx];
    }
}

struct MatchResult {
    int64_t seed;
    int32_t chunkX;
    int32_t chunkZ;
};

__global__ void findPeaksKernel(
    const float* correlation, int32_t tileSize,
    int32_t patternWidth, int32_t patternHeight,
    float threshold, int64_t seed,
    int32_t tileOffsetX, int32_t tileOffsetZ,
    MatchResult* matches, uint32_t* matchCount, uint32_t maxMatches
) {
    int32_t localX = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t localZ = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (localX < tileSize - patternWidth + 1 && localZ < tileSize - patternHeight + 1) {
        float val = correlation[localZ * tileSize + localX];
        if (val >= threshold) {
            uint32_t idx = atomicAdd(matchCount, 1);
            if (idx < maxMatches) {
                matches[idx].seed = seed;
                matches[idx].chunkX = tileOffsetX + localX;
                matches[idx].chunkZ = tileOffsetZ + localZ;
            }
        }
    }
}

// Combined check: positive correlation >= threshold AND anti-correlation >= antiThreshold
__global__ void findPeaksCombinedKernel(
    const float* posCorrelation, const float* negCorrelation,
    int32_t tileSize, int32_t patternWidth, int32_t patternHeight,
    float posThreshold, float negThreshold, int64_t seed,
    int32_t tileOffsetX, int32_t tileOffsetZ,
    MatchResult* matches, uint32_t* matchCount, uint32_t maxMatches
) {
    int32_t localX = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t localZ = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (localX < tileSize - patternWidth + 1 && localZ < tileSize - patternHeight + 1) {
        float posVal = posCorrelation[localZ * tileSize + localX];
        float negVal = negCorrelation[localZ * tileSize + localX];
        
        if (posVal >= posThreshold && negVal >= negThreshold) {
            uint32_t idx = atomicAdd(matchCount, 1);
            if (idx < maxMatches) {
                matches[idx].seed = seed;
                matches[idx].chunkX = tileOffsetX + localX;
                matches[idx].chunkZ = tileOffsetZ + localZ;
            }
        }
    }
}

__global__ void complexMultiplyKernel(cufftComplex* a, const cufftComplex* b, int32_t size) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float ar = a[idx].x, ai = a[idx].y;
        float br = b[idx].x, bi = b[idx].y;
        a[idx].x = ar * br - ai * bi;
        a[idx].y = ar * bi + ai * br;
    }
}

__global__ void normalizeKernel(float* data, int32_t size, float scale) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] *= scale;
}

__global__ void generateSeedsKernel(uint64_t startIndex, uint64_t count, int64_t* seeds) {
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    uint64_t globalIdx = startIndex + idx;
    uint64_t upperIdx = globalIdx >> 17;
    uint64_t lowerBits = globalIdx & ((1ULL << 17) - 1);
    uint64_t upperBits = upperIdx * 10;
    
    if (upperBits >= (1ULL << 31)) { seeds[idx] = -1; return; }
    
    uint64_t advanced = (upperBits << 17) | lowerBits;
    uint64_t internal = ((advanced - LCG_ADD) & MASK_48) * LCG_MULT_INV & MASK_48;
    int64_t slimeSeed = static_cast<int64_t>(internal ^ LCG_MULT);
    seeds[idx] = (slimeSeed ^ XOR_CONST) & MASK_48;
}

// ============================================================================
// FFT Pattern Matcher
// ============================================================================

class TilePatternMatcher {
private:
    int32_t tileSize, fftSize, numSlots;
    int32_t patternWidth, patternHeight;
    int32_t mustSlimeCount, mustNotSlimeCount;
    bool hasAntiPattern;
    
    float** d_bitmaps;
    float** d_invertedBitmaps;  // Only allocated if hasAntiPattern
    float** d_correlations;
    float** d_antiCorrelations; // Only allocated if hasAntiPattern
    cufftComplex** d_bitmapFFTs;
    cufftComplex** d_workFFTs;
    
    cufftComplex* d_patternFFT;      // FFT of # pattern
    cufftComplex* d_antiPatternFFT;  // FFT of . pattern (only if needed)
    
    MatchResult* d_matches;
    uint32_t* d_matchCount;
    
    cufftHandle* plansR2C;
    cufftHandle* plansC2R;
    cudaStream_t* streams;
    
    std::vector<MatchResult> h_matches;
    dim3 blockSize, gridSize;
    int fftBlocks, corrBlocks;
    float scale;

    void preparePatternFFT(cufftComplex* d_fft, const Pattern& pattern, bool antiPattern) {
        std::vector<float> h_pattern(tileSize * tileSize, 0.0f);
        
        for (int32_t z = 0; z < pattern.height; z++) {
            for (int32_t x = 0; x < pattern.width; x++) {
                int8_t val = pattern.grid[z][x];
                bool include = antiPattern ? (val == -1) : (val == 1);
                if (include) {
                    int32_t px = (tileSize - x) % tileSize;
                    int32_t pz = (tileSize - z) % tileSize;
                    h_pattern[pz * tileSize + px] = 1.0f;
                }
            }
        }
        
        float* d_pattern;
        cudaMalloc(&d_pattern, tileSize * tileSize * sizeof(float));
        cudaMemcpy(d_pattern, h_pattern.data(), tileSize * tileSize * sizeof(float), cudaMemcpyHostToDevice);
        cufftExecR2C(plansR2C[0], d_pattern, d_fft);
        cudaDeviceSynchronize();
        cudaFree(d_pattern);
    }

public:
    TilePatternMatcher(int32_t size, const Pattern& pattern, int32_t slots = NUM_FFT_SLOTS) 
        : tileSize(size), numSlots(slots) {
        
        patternWidth = pattern.width;
        patternHeight = pattern.height;
        mustSlimeCount = pattern.mustSlimeCount;
        mustNotSlimeCount = pattern.mustNotSlimeCount;
        hasAntiPattern = pattern.hasAntiPattern;
        
        int32_t totalSize = tileSize * tileSize;
        fftSize = tileSize * (tileSize / 2 + 1);
        
        blockSize = dim3(16, 16);
        gridSize = dim3((tileSize + 15) / 16, (tileSize + 15) / 16);
        fftBlocks = (fftSize + 255) / 256;
        corrBlocks = (totalSize + 255) / 256;
        scale = 1.0f / totalSize;
        
        // Allocate arrays
        d_bitmaps = new float*[numSlots];
        d_correlations = new float*[numSlots];
        d_bitmapFFTs = new cufftComplex*[numSlots];
        d_workFFTs = new cufftComplex*[numSlots];
        plansR2C = new cufftHandle[numSlots];
        plansC2R = new cufftHandle[numSlots];
        streams = new cudaStream_t[numSlots];
        
        if (hasAntiPattern) {
            d_invertedBitmaps = new float*[numSlots];
            d_antiCorrelations = new float*[numSlots];
        } else {
            d_invertedBitmaps = nullptr;
            d_antiCorrelations = nullptr;
        }
        
        double vramGB = numSlots * (totalSize * sizeof(float) * (hasAntiPattern ? 4 : 2) + 
                                    fftSize * sizeof(cufftComplex) * 2) / 1e9;
        printf("Allocating %d FFT slots (%.1f GB VRAM)...\n", numSlots, vramGB);
        fflush(stdout);
        
        for (int i = 0; i < numSlots; i++) {
            cudaMalloc(&d_bitmaps[i], totalSize * sizeof(float));
            cudaMalloc(&d_correlations[i], totalSize * sizeof(float));
            cudaMalloc(&d_bitmapFFTs[i], fftSize * sizeof(cufftComplex));
            cudaMalloc(&d_workFFTs[i], fftSize * sizeof(cufftComplex));
            
            if (hasAntiPattern) {
                cudaMalloc(&d_invertedBitmaps[i], totalSize * sizeof(float));
                cudaMalloc(&d_antiCorrelations[i], totalSize * sizeof(float));
            }
            
            cufftPlan2d(&plansR2C[i], tileSize, tileSize, CUFFT_R2C);
            cufftPlan2d(&plansC2R[i], tileSize, tileSize, CUFFT_C2R);
            cudaStreamCreate(&streams[i]);
            cufftSetStream(plansR2C[i], streams[i]);
            cufftSetStream(plansC2R[i], streams[i]);
        }
        
        // Pattern FFTs
        cudaMalloc(&d_patternFFT, fftSize * sizeof(cufftComplex));
        preparePatternFFT(d_patternFFT, pattern, false);
        
        if (hasAntiPattern) {
            cudaMalloc(&d_antiPatternFFT, fftSize * sizeof(cufftComplex));
            preparePatternFFT(d_antiPatternFFT, pattern, true);
        } else {
            d_antiPatternFFT = nullptr;
        }
        
        cudaMalloc(&d_matches, MAX_MATCHES_PER_BATCH * sizeof(MatchResult));
        cudaMalloc(&d_matchCount, sizeof(uint32_t));
        h_matches.resize(MAX_MATCHES_PER_BATCH);
        
        printf("FFT matcher ready.\n");
        fflush(stdout);
    }
    
    ~TilePatternMatcher() {
        for (int i = 0; i < numSlots; i++) {
            cudaFree(d_bitmaps[i]);
            cudaFree(d_correlations[i]);
            cudaFree(d_bitmapFFTs[i]);
            cudaFree(d_workFFTs[i]);
            if (hasAntiPattern) {
                cudaFree(d_invertedBitmaps[i]);
                cudaFree(d_antiCorrelations[i]);
            }
            cufftDestroy(plansR2C[i]);
            cufftDestroy(plansC2R[i]);
            cudaStreamDestroy(streams[i]);
        }
        delete[] d_bitmaps;
        delete[] d_correlations;
        delete[] d_bitmapFFTs;
        delete[] d_workFFTs;
        delete[] plansR2C;
        delete[] plansC2R;
        delete[] streams;
        if (hasAntiPattern) {
            delete[] d_invertedBitmaps;
            delete[] d_antiCorrelations;
        }
        
        cudaFree(d_patternFFT);
        if (d_antiPatternFFT) cudaFree(d_antiPatternFFT);
        cudaFree(d_matches);
        cudaFree(d_matchCount);
    }
    
    void findPatternsBatch(
        const std::vector<int64_t>& seeds,
        int32_t tileOffsetX, int32_t tileOffsetZ,
        std::vector<MatchResult>& results
    ) {
        if (seeds.empty()) return;
        
        cudaMemset(d_matchCount, 0, sizeof(uint32_t));
        
        float posThreshold = mustSlimeCount - 0.5f;
        float negThreshold = mustNotSlimeCount - 0.5f;
        int32_t totalSize = tileSize * tileSize;
        
        for (size_t batch = 0; batch < seeds.size(); batch += numSlots) {
            size_t batchSize = std::min((size_t)numSlots, seeds.size() - batch);
            
            // Generate bitmaps
            for (size_t i = 0; i < batchSize; i++) {
                generateTileBitmapKernel<<<gridSize, blockSize, 0, streams[i]>>>(
                    seeds[batch + i], d_bitmaps[i], tileSize, tileOffsetX, tileOffsetZ
                );
            }
            
            // Generate inverted bitmaps if needed
            if (hasAntiPattern) {
                for (size_t i = 0; i < batchSize; i++) {
                    generateInvertedBitmapKernel<<<corrBlocks, 256, 0, streams[i]>>>(
                        d_bitmaps[i], d_invertedBitmaps[i], totalSize
                    );
                }
            }
            
            // Forward FFT
            for (size_t i = 0; i < batchSize; i++) {
                cufftExecR2C(plansR2C[i], d_bitmaps[i], d_bitmapFFTs[i]);
            }
            
            // Positive pattern correlation
            for (size_t i = 0; i < batchSize; i++) {
                cudaMemcpyAsync(d_workFFTs[i], d_bitmapFFTs[i],
                               fftSize * sizeof(cufftComplex),
                               cudaMemcpyDeviceToDevice, streams[i]);
            }
            for (size_t i = 0; i < batchSize; i++) {
                complexMultiplyKernel<<<fftBlocks, 256, 0, streams[i]>>>(
                    d_workFFTs[i], d_patternFFT, fftSize);
            }
            for (size_t i = 0; i < batchSize; i++) {
                cufftExecC2R(plansC2R[i], d_workFFTs[i], d_correlations[i]);
            }
            for (size_t i = 0; i < batchSize; i++) {
                normalizeKernel<<<corrBlocks, 256, 0, streams[i]>>>(
                    d_correlations[i], totalSize, scale);
            }
            
            // Anti-pattern correlation (only if needed)
            if (hasAntiPattern) {
                // FFT of inverted bitmaps
                for (size_t i = 0; i < batchSize; i++) {
                    cufftExecR2C(plansR2C[i], d_invertedBitmaps[i], d_bitmapFFTs[i]);
                }
                for (size_t i = 0; i < batchSize; i++) {
                    cudaMemcpyAsync(d_workFFTs[i], d_bitmapFFTs[i],
                                   fftSize * sizeof(cufftComplex),
                                   cudaMemcpyDeviceToDevice, streams[i]);
                }
                for (size_t i = 0; i < batchSize; i++) {
                    complexMultiplyKernel<<<fftBlocks, 256, 0, streams[i]>>>(
                        d_workFFTs[i], d_antiPatternFFT, fftSize);
                }
                for (size_t i = 0; i < batchSize; i++) {
                    cufftExecC2R(plansC2R[i], d_workFFTs[i], d_antiCorrelations[i]);
                }
                for (size_t i = 0; i < batchSize; i++) {
                    normalizeKernel<<<corrBlocks, 256, 0, streams[i]>>>(
                        d_antiCorrelations[i], totalSize, scale);
                }
                
                // Combined peak detection
                for (size_t i = 0; i < batchSize; i++) {
                    findPeaksCombinedKernel<<<gridSize, blockSize, 0, streams[i]>>>(
                        d_correlations[i], d_antiCorrelations[i],
                        tileSize, patternWidth, patternHeight,
                        posThreshold, negThreshold, seeds[batch + i],
                        tileOffsetX, tileOffsetZ,
                        d_matches, d_matchCount, MAX_MATCHES_PER_BATCH
                    );
                }
            } else {
                // Simple peak detection (no anti-pattern)
                for (size_t i = 0; i < batchSize; i++) {
                    findPeaksKernel<<<gridSize, blockSize, 0, streams[i]>>>(
                        d_correlations[i], tileSize, patternWidth, patternHeight,
                        posThreshold, seeds[batch + i],
                        tileOffsetX, tileOffsetZ,
                        d_matches, d_matchCount, MAX_MATCHES_PER_BATCH
                    );
                }
            }
            
            // Sync
            for (size_t i = 0; i < batchSize; i++) {
                cudaStreamSynchronize(streams[i]);
            }
        }
        
        // Copy results
        uint32_t matchCount;
        cudaMemcpy(&matchCount, d_matchCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        matchCount = std::min(matchCount, MAX_MATCHES_PER_BATCH);
        
        if (matchCount > 0) {
            cudaMemcpy(h_matches.data(), d_matches, matchCount * sizeof(MatchResult), cudaMemcpyDeviceToHost);
            for (uint32_t i = 0; i < matchCount; i++) {
                results.push_back(h_matches[i]);
            }
        }
    }
    
    int32_t getTileSize() const { return tileSize; }
};

// ============================================================================
// Seed Generator
// ============================================================================

class SeedGenerator {
private:
    uint64_t currentIndex, totalSeeds;
    int64_t* d_seeds[NUM_CUDA_STREAMS];
    cudaStream_t streams[NUM_CUDA_STREAMS];

public:
    SeedGenerator() : currentIndex(0) {
        totalSeeds = ((1ULL << 31) / 10) * (1ULL << 17);
        for (int i = 0; i < NUM_CUDA_STREAMS; i++) {
            cudaMalloc(&d_seeds[i], SEED_BATCH_SIZE * sizeof(int64_t));
            cudaStreamCreate(&streams[i]);
        }
    }
    
    ~SeedGenerator() {
        for (int i = 0; i < NUM_CUDA_STREAMS; i++) {
            cudaFree(d_seeds[i]);
            cudaStreamDestroy(streams[i]);
        }
    }
    
    void reset() { currentIndex = 0; }
    uint64_t getTotal() const { return totalSeeds; }
    uint64_t getCurrent() const { return currentIndex; }
    bool done() const { return currentIndex >= totalSeeds; }
    cudaStream_t getStream(int idx) { return streams[idx % NUM_CUDA_STREAMS]; }
    
    int64_t* generateBatchGPU(uint64_t maxCount, uint64_t& actualCount, int streamIdx = 0) {
        actualCount = std::min(maxCount, totalSeeds - currentIndex);
        if (actualCount == 0) return nullptr;
        
        int threads = 256;
        int blocks = (actualCount + threads - 1) / threads;
        int sid = streamIdx % NUM_CUDA_STREAMS;
        
        generateSeedsKernel<<<blocks, threads, 0, streams[sid]>>>(currentIndex, actualCount, d_seeds[sid]);
        currentIndex += actualCount;
        return d_seeds[sid];
    }
};

// ============================================================================
// Result Storage
// ============================================================================

class ResultStorage {
private:
    std::mutex mtx;
    std::ofstream outFile;
    std::atomic<uint64_t> totalMatches{0};
    std::atomic<uint64_t> seedsProcessed{0};

public:
    ResultStorage(const char* filename = "slime_results.txt") {
        outFile.open(filename, std::ios::app);
        if (outFile.is_open()) {
            outFile << "# Pattern Search Results\n";
            outFile << "# Format: chunkX,chunkZ,seed\n";
            outFile.flush();
        }
    }
    
    ~ResultStorage() { if (outFile.is_open()) outFile.close(); }
    
    void addResults(const std::vector<MatchResult>& matches) {
        if (matches.empty()) return;
        std::lock_guard<std::mutex> lock(mtx);
        for (const auto& m : matches) {
            totalMatches++;
            if (outFile.is_open()) {
                outFile << m.chunkX << "," << m.chunkZ << "," << m.seed << "\n";
            }
        }
        outFile.flush();
    }
    
    void addSeedsProcessed(uint64_t count) { seedsProcessed += count; }
    uint64_t getMatchCount() const { return totalMatches.load(); }
    uint64_t getSeedsProcessed() const { return seedsProcessed.load(); }
};

// ============================================================================
// Main Search
// ============================================================================

void runSearch() {
    printf("=== Ultra Slime Chunk Finder ===\n");
    printf("=== User-Defined Pattern Search ===\n\n");
    
    // Load pattern
    if (!g_pattern.load("pattern.txt")) {
        printf("Failed to load pattern.txt\n");
        printf("Create pattern.txt with:\n");
        printf("  # = must be slime\n");
        printf("  . = must NOT be slime\n");
        printf("  X = don't care\n");
        printf("\nExample 3x3:\n###\n###\n###\n");
        return;
    }
    
    printf("Loaded pattern:\n");
    g_pattern.print();
    printf("\n");
    
    // Check CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) { printf("No CUDA devices!\n"); return; }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%.1f GB)\n", prop.name, prop.totalGlobalMem / 1e9);
    printf("Tile: %dx%d chunks\n\n", TILE_SIZE, TILE_SIZE);
    
    TilePatternMatcher matcher(TILE_SIZE, g_pattern);
    SeedGenerator seedGen;
    ResultStorage results;
    SpiralIterator tileIter;
    
    printf("Total seeds: %.2f trillion\n", seedGen.getTotal() / 1e12);
    printf("Press Ctrl+C to stop.\n\n");
    
    while (true) {
        int32_t minX, minZ, maxX, maxZ;
        tileIter.getChunkBounds(minX, minZ, maxX, maxZ);
        
        printf("\n--- Tile (%d,%d) chunks (%d,%d)-(%d,%d) ---\n",
               tileIter.tileX(), tileIter.tileZ(), minX, minZ, maxX, maxZ);
        fflush(stdout);
        
        seedGen.reset();
        uint64_t tileSeeds = 0, tileMatches = 0;
        auto tileStart = std::chrono::high_resolution_clock::now();
        
        int streamIdx = 0;
        while (!seedGen.done()) {
            uint64_t batchCount;
            int64_t* d_seeds = seedGen.generateBatchGPU(SEED_BATCH_SIZE, batchCount, streamIdx);
            if (!d_seeds || batchCount == 0) break;
            
            cudaStreamSynchronize(seedGen.getStream(streamIdx));
            
            // Copy seeds to host for batch processing
            std::vector<int64_t> h_seeds(batchCount);
            cudaMemcpy(h_seeds.data(), d_seeds, batchCount * sizeof(int64_t), cudaMemcpyDeviceToHost);
            
            // Remove invalid seeds
            h_seeds.erase(std::remove_if(h_seeds.begin(), h_seeds.end(), 
                          [](int64_t s) { return s < 0; }), h_seeds.end());
            
            std::vector<MatchResult> batchResults;
            matcher.findPatternsBatch(h_seeds, minX, minZ, batchResults);
            
            results.addResults(batchResults);
            results.addSeedsProcessed(batchCount);
            tileSeeds += batchCount;
            tileMatches += batchResults.size();
            streamIdx = (streamIdx + 1) % NUM_CUDA_STREAMS;
            
            // Progress
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - tileStart).count();
            double rate = tileSeeds / elapsed;
            double pct = 100.0 * seedGen.getCurrent() / seedGen.getTotal();
            
            printf("\r  [%.2f%%] %.2fB/s | Matches: %lu   ",
                   pct, rate / 1e9, (unsigned long)results.getMatchCount());
            fflush(stdout);
        }
        
        auto tileEnd = std::chrono::high_resolution_clock::now();
        double tileTime = std::chrono::duration<double>(tileEnd - tileStart).count();
        printf("\n  Tile done: %.1fs, %lu matches\n", tileTime, (unsigned long)tileMatches);
        
        tileIter.next();
    }
}

// ============================================================================
// Test
// ============================================================================

int runTest() {
    printf("=== Pattern Matcher Test ===\n\n");
    
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Test known 3x3 at seed 413563856, chunks (1495-1497, 8282-8284)
    int64_t seed = 413563856;
    printf("Testing seed %lld for 3x3 at (1495, 8282):\n", (long long)seed);
    
    // Verify chunks
    int passed = 0;
    for (int z = 8282; z <= 8284; z++) {
        for (int x = 1495; x <= 1497; x++) {
            if (isSlimeChunkHost(seed, x, z)) passed++;
        }
    }
    printf("Slime chunks: %d/9\n", passed);
    
    // Create test pattern (3x3 all slime)
    Pattern testPattern;
    testPattern.grid = {{1,1,1}, {1,1,1}, {1,1,1}};
    testPattern.width = 3;
    testPattern.height = 3;
    testPattern.mustSlimeCount = 9;
    testPattern.mustNotSlimeCount = 0;
    testPattern.hasAntiPattern = false;
    
    printf("\nTest pattern:\n");
    testPattern.print();
    
    // Small tile matcher
    TilePatternMatcher matcher(256, testPattern, 1);
    
    std::vector<int64_t> seeds = {seed};
    std::vector<MatchResult> results;
    matcher.findPatternsBatch(seeds, 1400, 8200, results);
    
    printf("\nFound %zu matches:\n", results.size());
    bool found = false;
    for (const auto& m : results) {
        printf("  (%d, %d)\n", m.chunkX, m.chunkZ);
        if (m.chunkX == 1495 && m.chunkZ == 8282) found = true;
    }
    
    printf("\nTest %s\n", (passed == 9 && found) ? "PASSED" : "FAILED");
    return (passed == 9 && found) ? 0 : 1;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    printf("Slime Pattern Finder\n");
    fflush(stdout);
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test") == 0) return runTest();
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--test] [-h]\n", argv[0]);
            printf("\nCreate pattern.txt with:\n");
            printf("  # = must be slime\n");
            printf("  . = must NOT be slime\n");
            printf("  X = don't care\n");
            return 0;
        }
    }
    
    runSearch();
    return 0;
}
