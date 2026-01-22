// Quick test to verify we find known 4x4 pattern at (64, 333)
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

constexpr uint64_t MASK_48 = (1ULL << 48) - 1;
constexpr uint64_t LCG_MULT = 0x5DEECE66DULL;
constexpr uint64_t LCG_ADD = 0xBULL;
constexpr uint64_t XOR_CONST = 0x3ad8025fULL;
constexpr uint64_t COMBINED_XOR = XOR_CONST ^ LCG_MULT;

constexpr int32_t SLIME_A = 0x4c1906;
constexpr int32_t SLIME_B = 0x5ac0db;
constexpr int64_t SLIME_C = 0x4307a7LL;
constexpr int32_t SLIME_D = 0x5f24f;

int64_t computePositionTerm(int32_t x, int32_t z) {
    int32_t xTerm = x * (x * SLIME_A + SLIME_B);
    int64_t zTerm = (int64_t)(z * z) * SLIME_C + z * SLIME_D;
    return (int64_t)xTerm + zTerm;
}

bool isSlime(int64_t seed, int32_t x, int32_t z) {
    int64_t posTerm = computePositionTerm(x, z);
    int64_t slimeSeed = seed + posTerm;
    uint64_t internal = ((uint64_t)slimeSeed ^ COMBINED_XOR) & MASK_48;
    uint64_t advanced = (internal * LCG_MULT + LCG_ADD) & MASK_48;
    return (advanced >> 17) % 10 == 0;
}

bool check4x4(int64_t seed, int32_t baseX, int32_t baseZ) {
    for (int dz = 0; dz < 4; dz++) {
        for (int dx = 0; dx < 4; dx++) {
            if (!isSlime(seed, baseX + dx, baseZ + dz)) return false;
        }
    }
    return true;
}

int main() {
    // Known result: position (64, 333), seed 100147049537412
    int32_t x = 64, z = 333;
    uint64_t seed = 100147049537412ULL;
    
    printf("Verifying known 4x4 pattern:\n");
    printf("  Position: (%d, %d)\n", x, z);
    printf("  Seed: %lu\n", (unsigned long)seed);
    
    if (check4x4(seed, x, z)) {
        printf("  Result: VALID 4x4 pattern!\n\n");
        
        printf("  Pattern visualization:\n");
        for (int dz = 0; dz < 4; dz++) {
            printf("  ");
            for (int dx = 0; dx < 4; dx++) {
                printf("%s ", isSlime(seed, x+dx, z+dz) ? "##" : "..");
            }
            printf("\n");
        }
    } else {
        printf("  Result: NOT a valid pattern\n");
    }
    
    return 0;
}
