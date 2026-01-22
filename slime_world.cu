/**
 * CUDA Slime Pattern Finder for a Given Seed
 * 
 * Searches for user-defined slime chunk patterns at all positions for a given seed.
 * 
 * Pattern format (loaded from pattern.txt or specified via --pattern):
 *   # = must be slime
 *   . = must NOT be slime  
 *   X = don't care (wildcard)
 * 
 * Example patterns:
 *   4x4 solid:     ####
 *                  ####
 *                  ####
 *                  ####
 * 
 *   Cross shape:   X#X
 *                  ###
 *                  X#X
 * 
 *   Ring:          ####
 *                  #..#
 *                  #..#
 *                  ####
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>

// ============================================================================
// LCG Constants
// ============================================================================

constexpr uint64_t MASK_32 = (1ULL << 32) - 1;
constexpr uint64_t MASK_48 = (1ULL << 48) - 1;
constexpr uint64_t LCG_MULT = 0x5DEECE66DULL;
constexpr uint64_t LCG_ADD = 0xBULL;
constexpr uint64_t COMBINED_XOR = 0x3ad8025fULL ^ LCG_MULT;

// Slime formula constants
constexpr int64_t SLIME_A = 0x4c1906;   // 4987142
constexpr int64_t SLIME_B = 0x5ac0db;   // 5947611
constexpr int64_t SLIME_C = 0x4307a7;   // 4392871
constexpr int64_t SLIME_D = 0x5f24f;    // 389711

constexpr uint32_t MAX_RESULTS = 1U << 20;
constexpr int32_t MAX_PATTERN_SIZE = 32;  // Maximum pattern dimension
constexpr double DEFAULT_FLUSH_INTERVAL = 10.0;  // Default flush interval in seconds

// ============================================================================
// Pattern Definition
// ============================================================================

struct Pattern {
    int8_t grid[MAX_PATTERN_SIZE][MAX_PATTERN_SIZE];  // 1=slime, -1=not slime, 0=don't care
    int32_t width;
    int32_t height;
    int32_t mustSlimeCount;
    int32_t mustNotSlimeCount;
    int32_t dontCareCount;
    
    Pattern() : width(0), height(0), mustSlimeCount(0), mustNotSlimeCount(0), dontCareCount(0) {
        memset(grid, 0, sizeof(grid));
    }
    
    bool load(const char* filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            printf("Error: Cannot open pattern file '%s'\n", filename);
            return false;
        }
        
        std::string line;
        height = 0;
        width = 0;
        mustSlimeCount = 0;
        mustNotSlimeCount = 0;
        dontCareCount = 0;
        
        while (std::getline(file, line) && height < MAX_PATTERN_SIZE) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '/' || line[0] == ';' || line[0] == '\r') continue;
            
            int32_t rowWidth = 0;
            for (char c : line) {
                if (rowWidth >= MAX_PATTERN_SIZE) break;
                
                if (c == '#') {
                    grid[height][rowWidth++] = 1;
                    mustSlimeCount++;
                } else if (c == '.') {
                    grid[height][rowWidth++] = -1;
                    mustNotSlimeCount++;
                } else if (c == 'X' || c == 'x' || c == ' ' || c == '?') {
                    grid[height][rowWidth++] = 0;
                    dontCareCount++;
                }
                // Ignore other characters
            }
            
            if (rowWidth > 0) {
                width = std::max(width, rowWidth);
                height++;
            }
        }
        
        // Pad rows to uniform width (with don't care)
        for (int z = 0; z < height; z++) {
            for (int x = 0; x < width; x++) {
                if (grid[z][x] == 0 && x >= width) {
                    dontCareCount++;
                }
            }
        }
        
        return height > 0 && width > 0;
    }
    
    // Create a solid NxN pattern (all must be slime)
    void createSolid(int32_t size) {
        width = height = size;
        mustSlimeCount = size * size;
        mustNotSlimeCount = 0;
        dontCareCount = 0;
        for (int z = 0; z < size; z++) {
            for (int x = 0; x < size; x++) {
                grid[z][x] = 1;
            }
        }
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
        printf("  Must be slime (#): %d\n", mustSlimeCount);
        printf("  Must NOT be slime (.): %d\n", mustNotSlimeCount);
        printf("  Don't care (X): %d\n", dontCareCount);
    }
};

// Global pattern (will be copied to GPU constant memory)
Pattern g_pattern;

// Device constant memory for pattern
__constant__ int8_t d_patternGrid[MAX_PATTERN_SIZE][MAX_PATTERN_SIZE];
__constant__ int32_t d_patternWidth;
__constant__ int32_t d_patternHeight;

// ============================================================================
// Host: Position term computation (separable)
// ============================================================================

inline int32_t to_signed_32(int64_t x) {
    x = x & MASK_32;
    if (x >= (1LL << 31)) x -= (1LL << 32);
    return (int32_t)x;
}

inline int64_t compute_f(int64_t x) {
    // f(x) = x * ((x*A + B) mod 2^32) with 32-bit overflow
    int32_t inner = to_signed_32(x * SLIME_A + SLIME_B);
    int32_t xTerm = to_signed_32(x * inner);
    return xTerm;
}

inline int64_t compute_g(int64_t z) {
    // g(z) = z^2 * C + z * D (64-bit arithmetic)
    return z * z * SLIME_C + z * SLIME_D;
}

inline int64_t compute_position_term(int64_t x, int64_t z) {
    return compute_f(x) + compute_g(z);
}

// ============================================================================
// Device: Slime chunk check
// ============================================================================

__device__ __forceinline__
int32_t d_to_signed_32(int64_t x) {
    x = x & MASK_32;
    if (x >= (1LL << 31)) x -= (1LL << 32);
    return (int32_t)x;
}

__device__ __forceinline__
int64_t d_compute_f(int64_t x) {
    int32_t inner = d_to_signed_32(x * SLIME_A + SLIME_B);
    int32_t xTerm = d_to_signed_32(x * inner);
    return xTerm;
}

__device__ __forceinline__
int64_t d_compute_g(int64_t z) {
    return z * z * SLIME_C + z * SLIME_D;
}

__device__ __forceinline__
bool isSlimeChunk(uint64_t seed, int64_t x, int64_t z) {
    int64_t pos_term = d_compute_f(x) + d_compute_g(z);
    uint64_t slime_seed = ((uint64_t)((int64_t)seed + pos_term)) & MASK_48;
    uint64_t internal = (slime_seed ^ COMBINED_XOR) & MASK_48;
    uint64_t advanced = (internal * LCG_MULT + LCG_ADD) & MASK_48;
    return (advanced >> 17) % 10 == 0;
}

// Check pattern using constant memory
// Returns true if position matches pattern requirements
__device__ __forceinline__
bool checkPattern(uint64_t seed, int64_t baseX, int64_t baseZ) {
    for (int dz = 0; dz < d_patternHeight; dz++) {
        for (int dx = 0; dx < d_patternWidth; dx++) {
            int8_t requirement = d_patternGrid[dz][dx];
            if (requirement == 0) continue;  // Don't care
            
            bool isSlime = isSlimeChunk(seed, baseX + dx, baseZ + dz);
            
            if (requirement == 1 && !isSlime) return false;  // Must be slime but isn't
            if (requirement == -1 && isSlime) return false;  // Must NOT be slime but is
        }
    }
    return true;
}

// Quick first-cell filter - check first required slime cell
__device__ __forceinline__
bool checkFirstCell(uint64_t seed, int64_t baseX, int64_t baseZ) {
    // Find first cell that must be slime and check it
    for (int dz = 0; dz < d_patternHeight; dz++) {
        for (int dx = 0; dx < d_patternWidth; dx++) {
            if (d_patternGrid[dz][dx] == 1) {
                return isSlimeChunk(seed, baseX + dx, baseZ + dz);
            }
        }
    }
    // No required slime cells - pattern matches trivially for first filter
    return true;
}

// ============================================================================
// Kernel: GPU brute force with first-cell filter
// ============================================================================

__global__ void searchBruteForce(
    uint64_t seed,
    int64_t minX, int64_t maxX,
    int64_t minZ, int64_t maxZ,
    uint64_t* results,
    uint32_t* resultCount,
    uint32_t maxResults
) {
    int64_t width = maxX - minX + 1;
    uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)width * (maxZ - minZ + 1);
    
    if (idx >= total) return;
    
    int64_t x = minX + (idx % width);
    int64_t z = minZ + (idx / width);
    
    // Quick first-cell filter (~10% pass rate for slime patterns)
    if (!checkFirstCell(seed, x, z)) return;
    
    // Full pattern check
    if (checkPattern(seed, x, z)) {
        uint32_t pos = atomicAdd(resultCount, 1);
        if (pos < maxResults) {
            results[pos] = ((uint64_t)(uint32_t)x) | (((uint64_t)(uint32_t)z) << 32);
        }
    }
}

// ============================================================================
// Host: Slime World Searcher
// ============================================================================

// Copy pattern to GPU constant memory
void copyPatternToDevice(const Pattern& pattern) {
    cudaMemcpyToSymbol(d_patternGrid, pattern.grid, sizeof(pattern.grid));
    cudaMemcpyToSymbol(d_patternWidth, &pattern.width, sizeof(int32_t));
    cudaMemcpyToSymbol(d_patternHeight, &pattern.height, sizeof(int32_t));
}

class SlimeWorldSearcher {
private:
    uint64_t seed;
    
    // GPU resources
    uint64_t* d_results;
    uint32_t* d_resultCount;
    
    // Output file for periodic writes
    std::ofstream* outFile;
    uint32_t lastWrittenCount;
    double flushInterval;
    
public:
    SlimeWorldSearcher(uint64_t seed_, const Pattern& pattern) : seed(seed_), outFile(nullptr), lastWrittenCount(0), flushInterval(DEFAULT_FLUSH_INTERVAL) {
        cudaMalloc(&d_results, MAX_RESULTS * sizeof(uint64_t));
        cudaMalloc(&d_resultCount, sizeof(uint32_t));
        
        // Copy pattern to GPU constant memory
        copyPatternToDevice(pattern);
    }
    
    ~SlimeWorldSearcher() {
        cudaFree(d_results);
        cudaFree(d_resultCount);
    }
    
    // Set output file for periodic writes
    void setOutputFile(std::ofstream* file) {
        outFile = file;
        lastWrittenCount = 0;
    }
    
    // Set flush interval in seconds
    void setFlushInterval(double seconds) {
        flushInterval = seconds;
    }
    
    // Write new results to file
    void flushNewResults() {
        if (!outFile || !outFile->is_open()) return;
        
        uint32_t currentCount;
        cudaMemcpy(&currentCount, d_resultCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        currentCount = std::min(currentCount, MAX_RESULTS);
        
        if (currentCount > lastWrittenCount) {
            // Copy only new results
            uint32_t newCount = currentCount - lastWrittenCount;
            std::vector<uint64_t> newResults(newCount);
            cudaMemcpy(newResults.data(), d_results + lastWrittenCount,
                      newCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            
            for (uint64_t packed : newResults) {
                int32_t x = (int32_t)(packed & 0xFFFFFFFF);
                int32_t z = (int32_t)(packed >> 32);
                int64_t dist = std::max(std::abs((int64_t)x), std::abs((int64_t)z));
                *outFile << x << "," << z << ","
                         << x * 16 << "," << z * 16 << ","
                         << dist << "\n";
            }
            outFile->flush();
            lastWrittenCount = currentCount;
        }
    }
    
    // Main search - uses optimized GPU brute force
    std::vector<std::pair<int64_t, int64_t>> search(int64_t radius) {
        int64_t minCoord = -radius;
        int64_t maxCoord = radius;
        int64_t width = maxCoord - minCoord + 1;
        uint64_t totalPositions = (uint64_t)width * width;
        
        printf("Search area: %lld x %lld = %llu positions\n\n",
               (long long)width, (long long)width, (unsigned long long)totalPositions);
        
        printf("Using GPU brute force with first-cell filter...\n\n");
        return searchBruteForceGPU(minCoord, maxCoord);
    }
    
    // GPU brute force with first-chunk filter (for smaller searches)
    std::vector<std::pair<int64_t, int64_t>> searchBruteForceGPU(int64_t minCoord, int64_t maxCoord) {
        std::vector<std::pair<int64_t, int64_t>> results;
        
        cudaMemset(d_resultCount, 0, sizeof(uint32_t));
        lastWrittenCount = 0;
        
        int64_t width = maxCoord - minCoord + 1;
        uint64_t totalPositions = (uint64_t)width * width;
        
        int threads = 256;
        
        // Process in row batches
        constexpr int64_t ROWS_PER_BATCH = 4096;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        auto lastFlushTime = startTime;
        uint64_t positionsChecked = 0;
        
        for (int64_t zStart = minCoord; zStart <= maxCoord; zStart += ROWS_PER_BATCH) {
            int64_t zEnd = std::min(zStart + ROWS_PER_BATCH - 1, maxCoord);
            int64_t batchHeight = zEnd - zStart + 1;
            
            uint64_t batchPositions = (uint64_t)width * batchHeight;
            int blocks = (batchPositions + threads - 1) / threads;
            
            searchBruteForce<<<blocks, threads>>>(
                seed,
                minCoord, maxCoord,
                zStart, zEnd,
                d_results,
                d_resultCount,
                MAX_RESULTS
            );
            
            positionsChecked += batchPositions;
            
            // Progress update
            cudaDeviceSynchronize();
            
            uint32_t currentResults;
            cudaMemcpy(&currentResults, d_resultCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - startTime).count();
            double rate = positionsChecked / std::max(elapsed, 0.001);
            double pct = 100.0 * positionsChecked / totalPositions;
            double eta = (totalPositions - positionsChecked) / rate;
            
            printf("\rProgress: %.1f%% | Checked: %.2fB | Rate: %.2fB/s | ETA: %.1fs | Found: %u    ",
                   pct, positionsChecked / 1e9, rate / 1e9, eta, currentResults);
            fflush(stdout);
            
            // Flush new results to file periodically
            double timeSinceFlush = std::chrono::duration<double>(now - lastFlushTime).count();
            if (timeSinceFlush >= flushInterval) {
                flushNewResults();
                lastFlushTime = now;
            }
        }
        
        // Final flush
        flushNewResults();
        
        printf("\n");
        
        // Get all results for return value
        uint32_t numResults;
        cudaMemcpy(&numResults, d_resultCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        numResults = std::min(numResults, MAX_RESULTS);
        
        if (numResults > 0) {
            std::vector<uint64_t> packedResults(numResults);
            cudaMemcpy(packedResults.data(), d_results,
                      numResults * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            
            for (uint64_t packed : packedResults) {
                int32_t x = (int32_t)(packed & 0xFFFFFFFF);
                int32_t z = (int32_t)(packed >> 32);
                results.push_back({x, z});
            }
        }
        
        return results;
    }
};

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* progname) {
    printf("Usage: %s <seed> [options]\n\n", progname);
    printf("Options:\n");
    printf("  --radius R      Search radius in chunks (default: 100000)\n");
    printf("  --pattern FILE  Load pattern from file (default: pattern.txt)\n");
    printf("  --size N        Use solid NxN pattern (default: 4)\n");
    printf("  --flush N       File flush interval in seconds (default: %.0f)\n", DEFAULT_FLUSH_INTERVAL);
    printf("\nPattern file format:\n");
    printf("  # = must be slime\n");
    printf("  . = must NOT be slime\n");
    printf("  X = don't care (wildcard)\n");
    printf("\nExamples:\n");
    printf("  %s 76198311894916 --radius 1000\n", progname);
    printf("  %s 76198311894916 --pattern mypattern.txt\n", progname);
    printf("  %s 76198311894916 --size 5 --radius 500000\n", progname);
    printf("  %s 76198311894916 --flush 60  # flush every 60 seconds\n", progname);
}

int main(int argc, char** argv) {
    printf("CUDA Slime World Pattern Finder\n");
    printf("===============================\n\n");
    
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    // Check for help
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }
    
    // Check CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%.1f GB, %d SMs)\n\n", prop.name,
           prop.totalGlobalMem / 1e9, prop.multiProcessorCount);
    
    // Parse arguments
    int64_t signedSeed = strtoll(argv[1], nullptr, 10);
    uint64_t seed = (uint64_t)signedSeed & MASK_48;
    int64_t searchRadius = 100000;
    const char* patternFile = nullptr;
    int solidSize = 0;
    double flushInterval = DEFAULT_FLUSH_INTERVAL;
    
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--radius") == 0 && i + 1 < argc) {
            searchRadius = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--pattern") == 0 && i + 1 < argc) {
            patternFile = argv[++i];
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            solidSize = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--flush") == 0 && i + 1 < argc) {
            flushInterval = atof(argv[++i]);
            if (flushInterval < 1.0) flushInterval = 1.0;  // Minimum 1 second
        }
    }
    
    // Load or create pattern
    if (solidSize > 0) {
        if (solidSize > MAX_PATTERN_SIZE) {
            printf("Error: Maximum pattern size is %d\n", MAX_PATTERN_SIZE);
            return 1;
        }
        g_pattern.createSolid(solidSize);
        printf("Using solid %dx%d pattern (all slime)\n\n", solidSize, solidSize);
    } else if (patternFile) {
        if (!g_pattern.load(patternFile)) {
            return 1;
        }
        printf("Loaded pattern from %s\n\n", patternFile);
    } else {
        // Try to load default pattern.txt, fall back to 4x4 solid
        if (g_pattern.load("pattern.txt")) {
            printf("Loaded pattern from pattern.txt\n\n");
        } else {
            g_pattern.createSolid(4);
            printf("Using default solid 4x4 pattern\n\n");
        }
    }
    
    g_pattern.print();
    printf("\n");
    
    printf("Seed: %lld (48-bit: %llu)\n", (long long)signedSeed, (unsigned long long)seed);
    printf("Search radius: %lld chunks\n\n", (long long)searchRadius);
    
    // Open output file early for periodic writes
    char filename[256];
    snprintf(filename, sizeof(filename), "slime_world_%llu_%dx%d.txt",
             (unsigned long long)seed, g_pattern.width, g_pattern.height);
    
    std::ofstream outFile(filename);
    outFile << "# Seed: " << seed << "\n";
    outFile << "# Pattern: " << g_pattern.width << "x" << g_pattern.height << "\n";
    outFile << "# Search radius: " << searchRadius << "\n";
    outFile << "# chunkX,chunkZ,blockX,blockZ,distance\n";
    outFile.flush();
    
    printf("Output file: %s (writing results periodically)\n\n", filename);
    
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    SlimeWorldSearcher searcher(seed, g_pattern);
    searcher.setOutputFile(&outFile);
    searcher.setFlushInterval(flushInterval);
    auto results = searcher.search(searchRadius);
    
    auto totalEnd = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(totalEnd - totalStart).count();
    
    outFile.close();
    
    printf("\nTotal time: %.2f seconds\n\n", totalTime);
    
    // Sort by distance for display
    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        return std::max(std::abs(a.first), std::abs(a.second)) <
               std::max(std::abs(b.first), std::abs(b.second));
    });
    
    // Display results
    printf("Found %zu patterns:\n", results.size());
    printf("----------------------------------------\n");
    
    int shown = 0;
    for (const auto& r : results) {
        int64_t dist = std::max(std::abs(r.first), std::abs(r.second));
        printf("  Chunk (%lld, %lld) | Block (%lld, %lld) | Distance: %lld\n",
               (long long)r.first, (long long)r.second,
               (long long)(r.first * 16), (long long)(r.second * 16),
               (long long)dist);
        if (++shown >= 100 && results.size() > 100) {
            printf("  ... and %zu more\n", results.size() - 100);
            break;
        }
    }
    printf("----------------------------------------\n");
    
    printf("\nResults saved to: %s\n", filename);
    
    return 0;
}
