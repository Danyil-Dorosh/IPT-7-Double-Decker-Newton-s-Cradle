# IPT Problem 7 — Double Decker Newton's Cradle

> **International Physics Tournament** | Problem 7
>
> *Make a double decker Newton's cradle (as in the video) and explain the phenomenon. Optimize the transmission of the kinetic energy between the levels. How does the number of balls per layer change the transmission efficiency?*
>
> Reference video: https://youtu.be/kY2YeM5fNDw

---

## Resources

| Resource | Link |
|---|---|
| Miro board | https://miro.com/app/board/uXjVGrgXBC8=/?share_link_id=291373337960 |
| Google Drive | https://drive.google.com/drive/folders/1Wp3sqertzj_x7-kZhxfvsWTb_3-Dbek4?usp=sharing |
| Overleaf (LaTeX notes) | https://www.overleaf.com/read/wcbpmkqygtjd#191235 |
| Notion workspace | https://www.notion.so/IPT-problem-7-32eaff79e3a3802383c1deced0b3eb5c?source=copy_link |
| Perplexity space | https://www.perplexity.ai/spaces/ipt-problem-7-newton-shit-hGAECJIeRiewwuyN6q2yLA |

---

## Repository Structure

```
.
├── notebooks/                          # Jupyter notebooks — simulations & data analysis
│   └── double_decker_cradle.ipynb      # Main simulation notebook
├── docs/                               # LaTeX notes, article PDFs, write-up drafts
├── data/                               # Experimental measurement data (CSV / JSON)
├── figures/                            # Generated plots and exported figures
└── README.md
```

---

## Physics Overview

### Standard Newton's Cradle

A Newton's cradle demonstrates conservation of momentum and kinetic energy through
elastic collisions. When *n* balls are released from one side, exactly *n* balls swing
out on the other side — this follows uniquely from the simultaneous conservation of
both momentum and kinetic energy.

### Double Decker Newton's Cradle

A double decker configuration adds a second row of balls suspended at a different height.
The collision between the two levels couples the two subsystems through a shared ball
(or a bridge element), enabling **energy transfer between levels**.

Key questions investigated:

1. What is the mechanism of inter-level energy transfer?
2. How can the geometry / mass ratio be optimized to maximize transmission efficiency η?
3. How does the number of balls *n* per layer affect η?

---

## Simulation

The main simulation is in [`notebooks/double_decker_cradle.ipynb`](notebooks/double_decker_cradle.ipynb).

It implements:

- **Elastic collision model** for an arbitrary number of balls per layer
- **Double-decker model** — two coupled layers, energy transmitted via a shared (bridge) ball
- **Efficiency sweep** — η as a function of the number of balls per layer *n*
- **Visualisation** of ball positions and energy distribution over time

### Running locally

```bash
pip install jupyter numpy scipy matplotlib
jupyter notebook notebooks/double_decker_cradle.ipynb
```

---

## Contributing

Add article PDFs to `docs/`, raw measurement data to `data/`, and export finished
figures to `figures/`. Keep notebooks reproducible (clear outputs before committing
when possible).
