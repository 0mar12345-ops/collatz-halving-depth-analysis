# Structural Drivers of Growth in Collatz Trajectories: A Quantitative Analysis of Halving Depth and Volatility

This repository hosts the computational framework, custom data engines, and statistical modeling scripts used to evaluate structural trajectory dynamics within the Collatz Conjecture across $1 \le n \le 100,000$.

##  Open-Access Publication
The formalized research paper detailing the underlying mathematical principles and algorithmic frameworks is archived and citable:
 **[Read the Preprint on Zenodo](https://zenodo.org)**

---

##  Repository Architecture

###  Algorithmic Implementations
* **`halving_depth_analysis.py`**  
  Executes log-linear trajectory tracking to evaluate numeric volatility against structural decay metrics. Computes ordinary least squares (OLS) linear regressions, custom 95% confidence intervals, standard errors, and discrete binned data aggregates (Generates Figures 3–4).
* **`parity_analysis.py`**  
  Uses programmatic bit-pattern masking via `defaultdict` hashing structures to isolate, classify, and bucket numeric sequences based on historical parity bit paths up to 10 iterations deep (Generates Table 1 and Figures 1–2).

###  Prerequisites & Setup
The analytical framework requires a standard 64-bit Python environment paired with foundational scientific computing packages:
```bash
pip install numpy matplotlib
```

---

##  Reproducibility & Validation
All datasets are generated algorithmically on-runtime. Running either core execution script will dynamically regenerate the exact mathematical plots, data bins, and frequency histograms detailed throughout the published text. 

*Note: Marginal floating-point variances may arise due to architecture-specific rounding behaviors during high-depth logarithmic scaling operations.*

---

##  Academic Attribution & Citation
**Author:** Omar Cardy  
**Institutional Affiliation:** Sorbonne University Abu Dhabi  
**Digital Object Identifier (DOI):** [10.5281/zenodo.21859804](https://doi.org)  
**Development Status:** Stable / Closed-Archive Pre-Term Verification  
