import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


# STANDARD Collatz (IMPORTANT: no forced halving)

def collatz_metrics(n):
    original = n
    steps = 0
    max_val = n
    parity_pattern = []

    while n != 1:
        if len(parity_pattern) < 10:
            parity_pattern.append(0 if n % 2 == 0 else 1)

        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1

        max_val = max(max_val, n)
        steps += 1

    # pad if needed
    while len(parity_pattern) < 10:
        parity_pattern.append(0)

    volatility = steps * (max_val / original)

    return steps, max_val, volatility, tuple(parity_pattern)


# DATA COLLECTION

groups = defaultdict(list)

for n in range(1, 100001):
    steps, max_val, vol, pattern = collatz_metrics(n)
    groups[pattern].append((steps, max_val, vol))


# TABLE 1

results = []

for key, values in groups.items():
    size = len(values)
    avg_steps = sum(v[0] for v in values) / size
    avg_max = sum(v[1] for v in values) / size
    avg_vol = sum(v[2] for v in values) / size

    results.append((key, size, avg_steps, avg_max, avg_vol))

results.sort(key=lambda x: -x[1])

print("\nTable 1: Top 10 Parity Groups\n")
for r in results[:10]:
    print(f"{r[0]}, Size={r[1]}, AvgSteps={r[2]:.2f}, AvgMax={r[3]:.0f}, AvgVol={r[4]:.0f}")


# FIGURE 1

top = results[:10]
volatility = [r[4] for r in top]

plt.figure()
plt.bar(range(1, 11), volatility)
plt.xlabel("Top 10 Parity Classes")
plt.ylabel("Average Volatility")
plt.title("Average Volatility by Parity Group")
plt.show()


# FIGURE 2

all_vol = [v[2] for group in groups.values() for v in group if v[2] > 0]
log_vol = np.log10(all_vol)

plt.figure()
plt.hist(log_vol, bins=50)
plt.xlabel("log10(Volatility)")
plt.ylabel("Frequency")
plt.title("Distribution of Volatility (Log Scale)")
plt.show()