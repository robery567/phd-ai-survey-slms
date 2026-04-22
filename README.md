# Safety and Alignment of Small Language Models: A Systematic Survey

This repository contains the analysis code, structured datasets, and supplementary materials for the survey paper:

> **Safety and Alignment of Small Language Models: A Systematic Survey of Methods, Evaluation, and Open Challenges**
> Robert-Mihai Colca, Dana Petcu
> West University of Timișoara, Romania

## Repository Structure

```
├── analysis/                    # Analysis scripts
│   ├── alignment_methods.py     # Alignment method comparison (Table 2)
│   ├── benchmark_analysis.py    # Benchmark gap analysis (Table 3)
│   ├── efficiency_safety.py     # Efficiency-safety interactions (Table 4)
│   └── bibliography_stats.py    # Citation statistics and temporal distribution
├── data/                        # Generated structured datasets
│   ├── alignment_methods.csv    # Alignment methods with properties
│   ├── alignment_methods.json   # Full method data with strengths/weaknesses
│   ├── benchmarks.json          # 19 safety benchmarks with metadata
│   └── efficiency_safety.json   # 9 efficiency techniques with safety impacts
├── figures/                     # Figure generation
│   ├── fig1-taxonomy.tex        # Taxonomy diagram (TikZ)
│   ├── fig2-timeline.py         # Timeline generation script
│   ├── fig2-timeline.pdf        # Generated timeline figure
│   └── fig2-timeline.png        # Generated timeline figure (raster)
├── main.tex                     # Survey manuscript (LaTeX)
├── references.bib               # Bibliography (157 entries)
└── README.md
```

## Running the Analysis

```bash
# Generate all structured datasets
python3 analysis/alignment_methods.py
python3 analysis/benchmark_analysis.py
python3 analysis/efficiency_safety.py

# Citation statistics
python3 analysis/bibliography_stats.py

# Regenerate timeline figure (requires matplotlib)
python3 figures/fig2-timeline.py
```

## Key Findings

The analysis scripts produce the following outputs:

- **Alignment method comparison**: 8 methods compared across compute, memory, SLM suitability, and robustness
- **Benchmark gap analysis**: Only 4/19 benchmarks have SLM-specific results; 0/19 are compression-aware; Romanian and Eastern European languages are uncovered
- **Efficiency-safety interactions**: 9 compression/efficiency techniques with documented safety impacts
- **Citation coverage**: 157 references, 26% from 2025-2026

## Citation

If you use this data or analysis in your work, please cite:

```bibtex
@article{colca2026slmsafety,
  title={Safety and Alignment of Small Language Models: A Systematic Survey of Methods, Evaluation, and Open Challenges},
  author={Colca, Robert-Mihai and Petcu, Dana},
  journal={Artificial Intelligence Review},
  year={2026}
}
```
