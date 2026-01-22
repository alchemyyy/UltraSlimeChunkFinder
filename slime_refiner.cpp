/**
 * Slime Refiner - Analyzes slime chunk patterns
 * 
 * Modes:
 *   --rect (default): Grows patterns into largest rectangles
 *   --glob: Finds largest contiguous slime regions (any shape)
 * 
 * Reads slime_results.txt and analyzes each pattern.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <cstring>
#include <iomanip>

// ============================================================================
// Constants (must match slime.cu)
// ============================================================================

constexpr uint64_t MASK_48 = (1ULL << 48) - 1;
constexpr uint64_t LCG_MULT = 0x5DEECE66DULL;
constexpr uint64_t LCG_ADD = 0xBULL;
constexpr uint64_t XOR_CONST = 0x3ad8025fULL;

constexpr int32_t SLIME_A = 0x4c1906;
constexpr int32_t SLIME_B = 0x5ac0db;
constexpr int64_t SLIME_C = 0x4307a7LL;
constexpr int32_t SLIME_D = 0x5f24f;

// ============================================================================
// Slime check (must match Java behavior)
// ============================================================================

inline bool isSlimeChunk(int64_t worldSeed, int32_t chunkX, int32_t chunkZ) {
    int32_t term1 = chunkX * chunkX * SLIME_A;
    int32_t term2 = chunkX * SLIME_B;
    int64_t term3 = (int64_t)(chunkZ * chunkZ) * SLIME_C;
    int32_t term4 = chunkZ * SLIME_D;

    int64_t slimeSeed = (worldSeed + term1 + term2 + term3 + term4) ^ XOR_CONST;
    uint64_t internal = (static_cast<uint64_t>(slimeSeed) ^ LCG_MULT) & MASK_48;
    uint64_t advanced = (internal * LCG_MULT + LCG_ADD) & MASK_48;
    return (advanced >> 17) % 10 == 0;
}

// ============================================================================
// Check if entire row is slime
// ============================================================================

bool isRowSlime(int64_t seed, int32_t minX, int32_t maxX, int32_t z) {
    for (int32_t x = minX; x <= maxX; x++) {
        if (!isSlimeChunk(seed, x, z)) return false;
    }
    return true;
}

// ============================================================================
// Check if entire column is slime
// ============================================================================

bool isColSlime(int64_t seed, int32_t x, int32_t minZ, int32_t maxZ) {
    for (int32_t z = minZ; z <= maxZ; z++) {
        if (!isSlimeChunk(seed, x, z)) return false;
    }
    return true;
}

// ============================================================================
// Grow rectangle from 3x3 starting point
// ============================================================================

struct Rectangle {
    int32_t minX, minZ, maxX, maxZ;
    int32_t width() const { return maxX - minX + 1; }
    int32_t height() const { return maxZ - minZ + 1; }
    int32_t size() const { return width() * height(); }
};

Rectangle growRectangle(int64_t seed, int32_t startX, int32_t startZ, int32_t patternSize = 3) {
    // Start with NxN pattern
    Rectangle rect = { startX, startZ, startX + patternSize - 1, startZ + patternSize - 1 };
    
    bool expanded = true;
    while (expanded) {
        expanded = false;
        
        // Try expand right (+X)
        if (isColSlime(seed, rect.maxX + 1, rect.minZ, rect.maxZ)) {
            rect.maxX++;
            expanded = true;
        }
        
        // Try expand left (-X)
        if (isColSlime(seed, rect.minX - 1, rect.minZ, rect.maxZ)) {
            rect.minX--;
            expanded = true;
        }
        
        // Try expand down (+Z)
        if (isRowSlime(seed, rect.minX, rect.maxX, rect.maxZ + 1)) {
            rect.maxZ++;
            expanded = true;
        }
        
        // Try expand up (-Z)
        if (isRowSlime(seed, rect.minX, rect.maxX, rect.minZ - 1)) {
            rect.minZ--;
            expanded = true;
        }
    }
    
    return rect;
}

// ============================================================================
// Glob (contiguous region) search using flood fill
// ============================================================================

struct Glob {
    int64_t seed;
    int32_t startX, startZ;      // Starting point
    int32_t minX, minZ, maxX, maxZ;  // Bounding box
    int32_t chunkCount;          // Total chunks in glob
    std::vector<std::pair<int32_t, int32_t>> chunks;  // All chunk coordinates
    
    int32_t boundingWidth() const { return maxX - minX + 1; }
    int32_t boundingHeight() const { return maxZ - minZ + 1; }
    int32_t boundingArea() const { return boundingWidth() * boundingHeight(); }
    double density() const { return (double)chunkCount / boundingArea(); }
};

// Hash function for coordinate pairs
struct CoordHash {
    size_t operator()(const std::pair<int32_t, int32_t>& p) const {
        return std::hash<int64_t>()(((int64_t)p.first << 32) | (uint32_t)p.second);
    }
};

// Flood fill to find contiguous slime region
// maxChunks limits search to prevent runaway on very large globs
Glob findGlob(int64_t seed, int32_t startX, int32_t startZ, int32_t maxChunks = 100000) {
    Glob glob;
    glob.seed = seed;
    glob.startX = startX;
    glob.startZ = startZ;
    glob.minX = glob.maxX = startX;
    glob.minZ = glob.maxZ = startZ;
    glob.chunkCount = 0;
    
    std::unordered_set<std::pair<int32_t, int32_t>, CoordHash> visited;
    std::queue<std::pair<int32_t, int32_t>> queue;
    
    queue.push({startX, startZ});
    visited.insert({startX, startZ});
    
    // 4-directional neighbors (can change to 8 for diagonal connectivity)
    const int32_t dx[] = {1, -1, 0, 0};
    const int32_t dz[] = {0, 0, 1, -1};
    
    while (!queue.empty() && glob.chunkCount < maxChunks) {
        auto [x, z] = queue.front();
        queue.pop();
        
        if (!isSlimeChunk(seed, x, z)) continue;
        
        // Add to glob
        glob.chunkCount++;
        glob.chunks.push_back({x, z});
        glob.minX = std::min(glob.minX, x);
        glob.maxX = std::max(glob.maxX, x);
        glob.minZ = std::min(glob.minZ, z);
        glob.maxZ = std::max(glob.maxZ, z);
        
        // Check neighbors
        for (int i = 0; i < 4; i++) {
            int32_t nx = x + dx[i];
            int32_t nz = z + dz[i];
            
            if (visited.find({nx, nz}) == visited.end()) {
                visited.insert({nx, nz});
                if (isSlimeChunk(seed, nx, nz)) {
                    queue.push({nx, nz});
                }
            }
        }
    }
    
    return glob;
}

// Find the largest inscribed rectangle in a glob (for practical use)
Rectangle findLargestRectInGlob(const Glob& glob) {
    if (glob.chunks.empty()) {
        return {0, 0, 0, 0};
    }
    
    // Build a set for O(1) lookup
    std::unordered_set<std::pair<int32_t, int32_t>, CoordHash> chunkSet(
        glob.chunks.begin(), glob.chunks.end());
    
    Rectangle best = {glob.chunks[0].first, glob.chunks[0].second,
                      glob.chunks[0].first, glob.chunks[0].second};
    
    // Try each chunk as potential top-left corner
    for (const auto& [sx, sz] : glob.chunks) {
        // Expand right as far as possible
        int32_t maxWidth = 1;
        while (chunkSet.count({sx + maxWidth, sz})) {
            maxWidth++;
        }
        
        // For each width, find max height
        int32_t currentMaxWidth = maxWidth;
        for (int32_t h = 1; h <= glob.boundingHeight(); h++) {
            // Check if row at sz + h - 1 is valid for current width
            int32_t validWidth = 0;
            for (int32_t w = 0; w < currentMaxWidth; w++) {
                if (chunkSet.count({sx + w, sz + h - 1})) {
                    validWidth++;
                } else {
                    break;
                }
            }
            currentMaxWidth = validWidth;
            if (currentMaxWidth == 0) break;
            
            // Check if this rectangle is better
            int32_t area = currentMaxWidth * h;
            if (area > best.size()) {
                best = {sx, sz, sx + currentMaxWidth - 1, sz + h - 1};
            }
        }
    }
    
    return best;
}

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* progname) {
    printf("Usage: %s [options] [input_file] [output_file]\n\n", progname);
    printf("Modes:\n");
    printf("  --rect       Grow patterns into largest rectangles (default)\n");
    printf("  --glob       Find largest contiguous slime regions (any shape)\n");
    printf("\nOptions:\n");
    printf("  --max N      Maximum chunks to explore per glob (default: 100000)\n");
    printf("  --pattern N  Starting pattern size NxN (default: 3)\n");
    printf("  -h, --help   Show this help\n");
    printf("\nInput file format: chunkX,chunkZ,seed (one per line)\n");
    printf("Default input: slime_results.txt\n");
    printf("Default output: refined_results.txt (rect) or glob_results.txt (glob)\n");
}

int main(int argc, char** argv) {
    const char* inputFile = "slime_results.txt";
    const char* outputFile = nullptr;
    bool globMode = false;
    int32_t maxGlobChunks = 100000;
    int32_t patternSize = 3;
    
    // Parse arguments
    std::vector<const char*> positionalArgs;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rect") == 0) {
            globMode = false;
        } else if (strcmp(argv[i], "--glob") == 0) {
            globMode = true;
        } else if (strcmp(argv[i], "--max") == 0 && i + 1 < argc) {
            maxGlobChunks = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pattern") == 0 && i + 1 < argc) {
            patternSize = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        } else if (argv[i][0] != '-') {
            positionalArgs.push_back(argv[i]);
        }
    }
    
    if (positionalArgs.size() >= 1) inputFile = positionalArgs[0];
    if (positionalArgs.size() >= 2) outputFile = positionalArgs[1];
    
    // Set default output file based on mode
    const char* defaultOutput = globMode ? "glob_results.txt" : "refined_results.txt";
    if (!outputFile) outputFile = defaultOutput;
    
    printf("Slime Refiner\n");
    printf("Mode: %s\n", globMode ? "GLOB (contiguous regions)" : "RECT (rectangles)");
    printf("Input: %s\n", inputFile);
    printf("Output: %s\n", outputFile);
    if (globMode) {
        printf("Max chunks per glob: %d\n", maxGlobChunks);
    }
    printf("Starting pattern size: %dx%d\n\n", patternSize, patternSize);
    
    std::ifstream in(inputFile);
    if (!in.is_open()) {
        printf("Error: Cannot open %s\n", inputFile);
        return 1;
    }
    
    std::ofstream out(outputFile);
    if (!out.is_open()) {
        printf("Error: Cannot open %s for writing\n", outputFile);
        return 1;
    }
    
    if (globMode) {
        out << "# Slime Glob Results - Largest contiguous regions\n";
        out << "# Format: globSize,boundingBox,density,largestRect,blockX,blockZ,seed\n";
    } else {
        out << "# Refined Slime Results - Largest rectangles grown from " << patternSize << "x" << patternSize << "\n";
        out << "# Format: size,length,width,chunkX(posX),chunkZ(posZ),seed\n";
    }
    
    std::string line;
    uint64_t processed = 0;
    uint64_t totalSize = 0;
    int32_t maxFound = 0;
    
    // For glob mode: track unique globs to avoid duplicates
    std::unordered_set<std::pair<int32_t, int32_t>, CoordHash> processedStarts;
    
    // Check for seed in header (slime_world format)
    int64_t headerSeed = 0;
    bool useHeaderSeed = false;
    
    while (std::getline(in, line)) {
        // Check for seed in header comment
        if (line.rfind("# Seed:", 0) == 0) {
            if (sscanf(line.c_str(), "# Seed: %lld", &headerSeed) == 1) {
                useHeaderSeed = true;
                printf("Found seed in header: %lld\n", (long long)headerSeed);
            }
            continue;
        }
        
        // Skip other comments and empty lines
        if (line.empty() || line[0] == '#') continue;
        
        // Try to parse different formats:
        // Format 1 (slime results): chunkX,chunkZ,seed
        // Format 2 (slime_world): chunkX,chunkZ,blockX,blockZ,distance
        int32_t chunkX, chunkZ;
        int64_t seed;
        int32_t blockX, blockZ, distance;
        
        if (sscanf(line.c_str(), "%d,%d,%lld", &chunkX, &chunkZ, &seed) == 3) {
            // Check if it's actually format 2 (5 fields)
            if (sscanf(line.c_str(), "%d,%d,%d,%d,%d", &chunkX, &chunkZ, &blockX, &blockZ, &distance) == 5) {
                // It's slime_world format - use header seed
                if (!useHeaderSeed) {
                    printf("Error: slime_world format detected but no seed in header\n");
                    continue;
                }
                seed = headerSeed;
            }
            // else it's format 1, seed already parsed
        } else {
            continue;  // Skip malformed lines
        }
        
        if (globMode) {
            // Glob mode: find contiguous region
            // First, find an actual slime chunk within the pattern bounds
            // (the reported position might be a corner with wildcards)
            int32_t startX = chunkX, startZ = chunkZ;
            bool foundStart = false;
            for (int32_t dz = 0; dz < patternSize && !foundStart; dz++) {
                for (int32_t dx = 0; dx < patternSize && !foundStart; dx++) {
                    if (isSlimeChunk(seed, chunkX + dx, chunkZ + dz)) {
                        startX = chunkX + dx;
                        startZ = chunkZ + dz;
                        foundStart = true;
                    }
                }
            }
            
            if (!foundStart) {
                // No slime chunk found in pattern area - skip
                continue;
            }
            
            Glob glob = findGlob(seed, startX, startZ, maxGlobChunks);
            
            // Find largest rectangle within the glob
            Rectangle bestRect = findLargestRectInGlob(glob);
            
            // Output format: globSize,boundingBox,density,largestRect,blockX,blockZ,seed
            char boundingBox[64], largestRect[64];
            snprintf(boundingBox, sizeof(boundingBox), "%dx%d", 
                     glob.boundingWidth(), glob.boundingHeight());
            snprintf(largestRect, sizeof(largestRect), "%dx%d@(%d,%d)",
                     bestRect.width(), bestRect.height(), bestRect.minX * 16, bestRect.minZ * 16);
            
            out << glob.chunkCount << ","
                << boundingBox << ","
                << std::fixed << glob.density() << ","
                << largestRect << ","
                << chunkX * 16 << "," << chunkZ * 16 << ","
                << seed;
            
            if (glob.chunkCount >= maxGlobChunks) {
                out << ",TRUNCATED";
            }
            out << "\n";
            
            processed++;
            totalSize += glob.chunkCount;
            
            if (glob.chunkCount > maxFound) {
                maxFound = glob.chunkCount;
                printf("New max glob: %d chunks (bbox %dx%d, density %.1f%%, best rect %dx%d) at (%d,%d) seed %lld%s\n",
                       glob.chunkCount, glob.boundingWidth(), glob.boundingHeight(),
                       glob.density() * 100, bestRect.width(), bestRect.height(),
                       chunkX, chunkZ, (long long)seed,
                       glob.chunkCount >= maxGlobChunks ? " [TRUNCATED]" : "");
            }
        } else {
            // Rectangle mode: grow the rectangle
            Rectangle rect = growRectangle(seed, chunkX, chunkZ, patternSize);
            
            // Output: size,length,width,chunkX(posX),chunkZ(posZ),seed
            int32_t length = std::max(rect.width(), rect.height());
            int32_t width = std::min(rect.width(), rect.height());
            int32_t posX = rect.minX * 16;
            int32_t posZ = rect.minZ * 16;
            
            out << rect.size() << ","
                << length << ","
                << width << ","
                << rect.minX << "(" << posX << "),"
                << rect.minZ << "(" << posZ << "),"
                << seed << "\n";
            
            processed++;
            totalSize += rect.size();
            
            if (rect.size() > maxFound) {
                maxFound = rect.size();
                printf("New max rect: %dx%d = %d chunks at (%d,%d) seed %lld\n",
                       rect.width(), rect.height(), rect.size(),
                       rect.minX, rect.minZ, (long long)seed);
            }
        }
        
        if (processed % 10000 == 0) {
            printf("Processed %lu entries, avg size: %.2f\n",
                   (unsigned long)processed, (double)totalSize / processed);
        }
    }
    
    in.close();
    out.close();
    
    printf("\nDone! Processed %lu entries\n", (unsigned long)processed);
    if (globMode) {
        printf("Average glob size: %.2f chunks\n", 
               processed > 0 ? (double)totalSize / processed : 0.0);
        printf("Largest glob found: %d chunks\n", maxFound);
    } else {
        printf("Average rectangle size: %.2f chunks\n", 
               processed > 0 ? (double)totalSize / processed : 0.0);
        printf("Largest rectangle found: %d chunks\n", maxFound);
    }
    printf("Results written to %s\n", outputFile);
    
    return 0;
}
