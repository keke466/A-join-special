# A Joint Spectral Perspective: Average-Case Analysis with Anisotropic Initialization

This repository contains the source code and LaTeX files for the paper:

**Average-Case Analysis with Anisotropic Initialization: A Joint Spectral Perspective**  
Guohui Zhang  
*Daqing Normal University, China*

The paper extends classical average-case analysis of first-order optimization methods to anisotropic initialization. Under a commutativity assumption between the Hessian and the initial covariance, we derive exact finite-dimensional expressions and high-dimensional limits in terms of their joint spectral distribution. Numerical experiments with gradient descent and Nesterov's accelerated method validate the theoretical predictions, showing that alignment between the eigenbases significantly affects convergence speed.

## Repository Structure

```
.
├── paper/                      # LaTeX source of the paper
│   ├── template.tex            # Main LaTeX file (Optimization Letters style)
│   ├── gd_nesterov_convergence_vertical.pdf   # Figure 1
│   ├── figure2_robustness_clean.pdf           # Figure 2
│   └── references.bib          # BibTeX references (if separate)
├── code/                        # Python code for experiments
│   ├── exp1_four_cases.py       # Experiment 1: four initialization types (d=50)
│   ├── exp2_angle_scan.py       # Experiment 2: 2D angle sweep and robustness
│   ├── requirements.txt         # Python dependencies
│   └── README_code.md           # Detailed code documentation (optional)
├── data/                         # Generated data (CSV files)
│   ├── four_cases_stats_seed_42.csv
│   ├── angle_scan_raw.csv
│   └── ...
└── README.md                     # This file
```

Requirements

To run the experiments, you need Python 3.8+ and the following packages:
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`

Install them via:
```bash
pip install -r code/requirements.txt
```

To compile the paper, a LaTeX distribution (TeX Live 2020+ or MiKTeX) with the `svjour3` class is required. The class is included in most distributions; otherwise, download it from [Springer's website](https://www.springer.com/gp/authors-editors/book-authors-editors/manuscript-preparation/5636).

Reproducing the Experiments

Experiment 1: Four Initialization Types (d=50)
Run:
```bash
python code/exp1_four_cases.py
```
This script performs 50 random initializations for each of the four cases (isotropic, independent, aligned, misaligned) and saves:
- Raw data: `four_cases_raw.csv`
- Summary statistics: `four_cases_stats.csv`

To replicate the robustness analysis with different random seeds, modify the `master_seed` variable in the script or run it multiple times manually.

Experiment 2: 2D Angle Sweep and Robustness
Run:
```bash
python code/exp2_angle_scan.py
```
This generates:
- Raw angle‑scan data: `angle_scan_raw.csv`
- Summary statistics: `angle_scan_stats.csv`
- Figure: `angle_vs_iteration.pdf` (optional)

The script also produces the data for Figure 2 (robustness bar chart) used in the paper.

Compiling the Paper

Navigate to the `paper/` directory and run:
```bash
pdflatex template.tex
bibtex template    # if using BibTeX
pdflatex template.tex
pdflatex template.tex
```
Make sure the two PDF figures are present in the same directory. The final output is `template.pdf`.

Results

The main numerical results are presented in Table 1 and Figures 1–2 of the paper. All intermediate data are saved as CSV files in the `data/` folder. You can reproduce the figures using the provided Python scripts or your own analysis tools.

Citing This Work

If you use this code or find the paper useful, please cite:

```bibtex
@article{Zhang2026Joint,
  title={Average-Case Analysis with Anisotropic Initialization: A Joint Spectral Perspective},
  author={Zhang, Guohui},
  journal={Optimization Letters},
  year={2026},
  note={To appear}
}

License

The code in this repository is released under the MIT License. See `LICENSE` for details. The LaTeX source is provided for transparency and may be used under the terms of the Springer copyright agreement.

Contact

For questions or comments, please contact Guohui Zhang at dqzgh@163.com.

---

*Last updated: February 2026*
