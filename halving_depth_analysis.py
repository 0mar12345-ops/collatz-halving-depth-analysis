import matplotlib.pyplot as plt
import numpy as np


# HALVING DEPTH VERSION

def collatz_metrics(n):
    original = n
    steps = 0
    max_val = n
    halving_counts = []

    while n != 1:
        if n % 2 == 1:
            n = 3 * n + 1

            count = 0
            while n % 2 == 0:
                n //= 2
                count += 1

            halving_counts.append(count)
        else:
            n //= 2

        max_val = max(max_val, n)
        steps += 1

    avg_halving = sum(halving_counts)/len(halving_counts) if halving_counts else 0
    volatility = steps * (max_val / original)

    return steps, max_val, volatility, avg_halving


# DATA

data = []

for n in range(1, 100001):
    steps, max_val, vol, avg_halving = collatz_metrics(n)
    data.append((steps, max_val, vol, avg_halving))


# PREP

clean_x = []
clean_y = []

for steps, max_val, vol, avg_halving in data:
    if vol > 0:
        clean_x.append(avg_halving)
        clean_y.append(np.log10(vol))

x = np.array(clean_x)
y = np.array(clean_y)


# FIGURE 3 (scatter)

plt.figure()
plt.scatter(x, y, s=1)
plt.xlabel("Average Halving Depth")
plt.ylabel("log10(Volatility)")
plt.title("Halving Depth vs Log Volatility")
plt.show()


# FIGURE 3 (trend line)

mask = (x >= 1.5) & (x <= 5.0)
x_fit = x[mask]
y_fit = y[mask]

m, b = np.polyfit(x_fit, y_fit, 1)

x_line = np.linspace(min(x_fit), max(x_fit), 100)
y_line = m * x_line + b

plt.figure()
plt.scatter(x, y, s=1)
plt.plot(x_line, y_line)
plt.xlabel("Average Halving Depth")
plt.ylabel("log10(Volatility)")
plt.title("Halving Depth vs Log Volatility (with Trend Line)")
plt.show()


# STATS

r = np.corrcoef(x_fit, y_fit)[0,1]
r_squared = r**2

print("Slope (m):", m)
print("Correlation (r):", r)
print("R^2:", r_squared)


# CONFIDENCE INTERVAL

n = len(x_fit)
y_pred = m * x_fit + b
residuals = y_fit - y_pred

se = np.sqrt(np.sum(residuals**2)/(n-2)) / np.sqrt(np.sum((x_fit - np.mean(x_fit))**2))

ci_lower = m - 1.96 * se
ci_upper = m + 1.96 * se

print("95% CI for slope:", (float(ci_lower), float(ci_upper)))


# FIGURE 4 (binned)

bins = np.linspace(min(x), max(x), 20)
bin_indices = np.digitize(x, bins)

bin_means = []
bin_centers = []

for i in range(1, len(bins)):
    vals = y[bin_indices == i]
    if len(vals) > 0:
        bin_means.append(np.mean(vals))
        bin_centers.append((bins[i] + bins[i-1])/2)

plt.figure()
plt.plot(bin_centers, bin_means)
plt.xlabel("Average Halving Depth")
plt.ylabel("Mean log10(Volatility)")
plt.title("Binned Relationship Between Halving Depth and Volatility")
plt.show()