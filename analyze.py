import numpy as np
from collections import Counter
import math

# Load data
positions = []
with open(r'C:\Users\Alchemy\Desktop\@workbench\UltraSlimeChunkFinder\slime_4x4_positions.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        x, z, count = int(parts[0]), int(parts[1]), int(parts[2])
        positions.append((x, z, count))

xs = np.array([p[0] for p in positions])
zs = np.array([p[1] for p in positions])
counts = np.array([p[2] for p in positions])

print(f"Total positions: {len(positions)}")
print(f"Total seeds: {counts.sum()}")

print(f"\n{'='*60}")
print("COORDINATE STATISTICS")
print('='*60)
print(f"X range: [{xs.min()}, {xs.max()}]")
print(f"Z range: [{zs.min()}, {zs.max()}]")
print(f"X mean: {xs.mean():.1f}, std: {xs.std():.1f}")
print(f"Z mean: {zs.mean():.1f}, std: {zs.std():.1f}")

print(f"\n{'='*60}")
print("SEED COUNT STATISTICS")
print('='*60)
print(f"Min seeds at position: {counts.min()}")
print(f"Max seeds at position: {counts.max()}")
print(f"Mean seeds per position: {counts.mean():.1f}")
print(f"Median seeds per position: {np.median(counts):.1f}")
print(f"Std dev: {counts.std():.1f}")

# Seed count distribution
print("\nSeed count distribution:")
bins = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
for i in range(len(bins)-1):
    c = np.sum((counts >= bins[i]) & (counts < bins[i+1]))
    print(f"  {bins[i]:5d}-{bins[i+1]:5d}: {c:4d} positions ({100*c/len(counts):.1f}%)")
c = np.sum(counts >= bins[-1])
print(f"  {bins[-1]:5d}+     : {c:4d} positions ({100*c/len(counts):.1f}%)")

print(f"\n{'='*60}")
print("DISTANCE FROM ORIGIN")
print('='*60)
distances = np.sqrt(xs**2 + zs**2)
print(f"Min distance: {distances.min():.1f}")
print(f"Max distance: {distances.max():.1f}")
print(f"Mean distance: {distances.mean():.1f}")
print(f"Median distance: {np.median(distances):.1f}")

# Radial distribution
print("\nRadial distribution (positions per 1000-unit ring):")
ring_width = 1000
ring_counts = {}
for d in distances:
    ring = int(d // ring_width) * ring_width
    ring_counts[ring] = ring_counts.get(ring, 0) + 1

for ring in sorted(ring_counts.keys()):
    area = math.pi * ((ring + ring_width)**2 - ring**2)
    density = ring_counts[ring] / (area / 1e6)
    print(f"  {ring:5d}-{ring+ring_width:5d}: {ring_counts[ring]:4d} positions (density: {density:.2f}/M)")

print(f"\n{'='*60}")
print("MODULAR PATTERNS")
print('='*60)
for mod in [2, 3, 4, 5, 8, 10, 16]:
    x_mod = {}
    z_mod = {}
    for x in xs:
        r = x % mod
        x_mod[r] = x_mod.get(r, 0) + 1
    for z in zs:
        r = z % mod
        z_mod[r] = z_mod.get(r, 0) + 1
    print(f"\nmod {mod}:")
    print(f"  X: {dict(sorted(x_mod.items()))}")
    print(f"  Z: {dict(sorted(z_mod.items()))}")
    
    # Chi-square test for uniformity
    expected = len(positions) / mod
    x_chi2 = sum((v - expected)**2 / expected for v in x_mod.values())
    z_chi2 = sum((v - expected)**2 / expected for v in z_mod.values())
    print(f"  X chi2: {x_chi2:.2f}, Z chi2: {z_chi2:.2f} (uniform if < {mod*3})")

print(f"\n{'='*60}")
print("QUADRANT DISTRIBUTION")
print('='*60)
q1 = np.sum((xs >= 0) & (zs >= 0))  # +X, +Z
q2 = np.sum((xs < 0) & (zs >= 0))   # -X, +Z
q3 = np.sum((xs < 0) & (zs < 0))    # -X, -Z
q4 = np.sum((xs >= 0) & (zs < 0))   # +X, -Z
print(f"Q1 (+X,+Z): {q1:4d} ({100*q1/len(positions):.1f}%)")
print(f"Q2 (-X,+Z): {q2:4d} ({100*q2/len(positions):.1f}%)")
print(f"Q3 (-X,-Z): {q3:4d} ({100*q3/len(positions):.1f}%)")
print(f"Q4 (+X,-Z): {q4:4d} ({100*q4/len(positions):.1f}%)")

print(f"\n{'='*60}")
print("CORRELATION ANALYSIS")
print('='*60)
# Correlation between x and z
corr_xz = np.corrcoef(xs, zs)[0,1]
print(f"X-Z correlation: {corr_xz:.4f}")

# Correlation between position and seed count
corr_x_count = np.corrcoef(xs, counts)[0,1]
corr_z_count = np.corrcoef(zs, counts)[0,1]
corr_dist_count = np.corrcoef(distances, counts)[0,1]
print(f"X-count correlation: {corr_x_count:.4f}")
print(f"Z-count correlation: {corr_z_count:.4f}")
print(f"Distance-count correlation: {corr_dist_count:.4f}")

print(f"\n{'='*60}")
print("CLUSTERING ANALYSIS")
print('='*60)
# Check for position clustering - nearest neighbor distances
from scipy.spatial import distance_matrix
coords = np.column_stack([xs, zs])

# Sample for efficiency
if len(coords) > 1000:
    sample_idx = np.random.choice(len(coords), 1000, replace=False)
    sample_coords = coords[sample_idx]
else:
    sample_coords = coords

dists = distance_matrix(sample_coords, sample_coords)
np.fill_diagonal(dists, np.inf)
nn_dists = dists.min(axis=1)
print(f"Nearest neighbor distances (sampled):")
print(f"  Min: {nn_dists.min():.1f}")
print(f"  Max: {nn_dists.max():.1f}")
print(f"  Mean: {nn_dists.mean():.1f}")
print(f"  Median: {np.median(nn_dists):.1f}")

# Expected NN distance for uniform random: ~0.5 * sqrt(Area/N)
area = (xs.max() - xs.min()) * (zs.max() - zs.min())
expected_nn = 0.5 * np.sqrt(area / len(positions))
print(f"  Expected (uniform): {expected_nn:.1f}")
print(f"  Ratio (actual/expected): {nn_dists.mean()/expected_nn:.2f}")
print("  (< 1 = clustered, > 1 = dispersed, ~1 = random)")

print(f"\n{'='*60}")
print("GCD PATTERNS IN COORDINATES")
print('='*60)
# Check if coordinates share common factors
def gcd(a, b):
    while b: a, b = b, a % b
    return abs(a)

# GCD of all x coordinates
x_gcd = xs[0]
for x in xs[1:]:
    x_gcd = gcd(x_gcd, x)
print(f"GCD of all X coordinates: {x_gcd}")

z_gcd = zs[0]
for z in zs[1:]:
    z_gcd = gcd(z_gcd, z)
print(f"GCD of all Z coordinates: {z_gcd}")

# Check x+z and x-z patterns
sums = xs + zs
diffs = xs - zs
sum_gcd = sums[0]
for s in sums[1:]:
    sum_gcd = gcd(sum_gcd, s)
diff_gcd = diffs[0]
for d in diffs[1:]:
    diff_gcd = gcd(diff_gcd, d)
print(f"GCD of (X+Z): {sum_gcd}")
print(f"GCD of (X-Z): {diff_gcd}")

print(f"\n{'='*60}")
print("TOP 20 POSITIONS BY SEED COUNT")
print('='*60)
sorted_pos = sorted(positions, key=lambda p: -p[2])
for i, (x, z, c) in enumerate(sorted_pos[:20]):
    print(f"  {i+1:2d}. ({x:6d}, {z:6d}): {c:5d} seeds")

print(f"\n{'='*60}")
print("LATTICE/GRID PATTERN DETECTION")
print('='*60)
# Check if positions fall on a lattice by looking at pairwise differences
x_diffs = []
z_diffs = []
for i in range(min(500, len(positions))):
    for j in range(i+1, min(500, len(positions))):
        x_diffs.append(abs(xs[i] - xs[j]))
        z_diffs.append(abs(zs[i] - zs[j]))

x_diff_counter = Counter(x_diffs)
z_diff_counter = Counter(z_diffs)

print("Most common X differences:")
for diff, cnt in sorted(x_diff_counter.items(), key=lambda x: -x[1])[:10]:
    print(f"  {diff}: {cnt} times")

print("\nMost common Z differences:")
for diff, cnt in sorted(z_diff_counter.items(), key=lambda x: -x[1])[:10]:
    print(f"  {diff}: {cnt} times")

# Check for periodicity
print("\nSmall factor analysis (differences mod small primes):")
for p in [2, 3, 5, 7, 11, 13]:
    x_mod_cnt = Counter([d % p for d in x_diffs if d > 0])
    z_mod_cnt = Counter([d % p for d in z_diffs if d > 0])
    print(f"  mod {p:2d} - X: {dict(sorted(x_mod_cnt.items()))}")
    print(f"         Z: {dict(sorted(z_mod_cnt.items()))}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
