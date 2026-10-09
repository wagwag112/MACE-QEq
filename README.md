# MACE-QEq Active Learning Framework for Metal | Solid Electrolyte Interfaces

Automated active-learning infrastructure for constant-potential atomistic simulations of metal–solid electrolyte ($\text{Metal} \mid \text{SE}$) interfaces.  
This implementation reproduces and extends the methodological framework established by **Hu et al. (*Nat. Commun.* 2025, 16:7379)** from liquid electrolytes ($EC + LiPF_6$) to solid-state battery electrolytes ($\text{Li} \mid \text{Li}_3\text{N}$ baseline), replacing DeePMD with equivariant **MACE** (Higher-Order Equivariant Message Passing Neural Networks) and VASP-driven density functional theory.

---

## 1. Theoretical Foundations

### 1.1 Energy Partitioning & Residual Learning
The total potential energy of an atomistic system is partitioned into short-range quantum-mechanical bonding interactions and long-range electrostatics:
$$E_{\text{Total}}(\mathbf{R}) = E_{\text{Short}}(\mathbf{R}) + E_{\text{QEq}}(\mathbf{R}, \mathbf{Q})$$

* **$E_{\text{Total}}$**: Ground-state DFT electronic energy computed via VASP ($\text{PBE} + \text{D3(BJ)}$).
* **$E_{\text{QEq}}$**: Long-range Coulombic and chemical-potential energy solved via Charge Equilibration (QEq) with Particle Mesh Ewald (PME).
* **$E_{\text{Short}}$**: Non-electrostatic residual ($E_{\text{DFT}} - E_{\text{QEq}}$) fitted by the MACE potential.

Corresponding atomic forces follow exact analytical derivatives:
```math
\mathbf{F}_i = -\nabla_{\mathbf{R}_i} E_{\text{Total}} = \mathbf{F}_i^{\text{Short}} + \mathbf{F}_i^{\text{QEq}}
```

### 1.2 QEq Energy Formulation
Under 3D periodic boundary conditions with slab geometry, $E_{\text{QEq}}$ is defined as:
$$E_{\text{QEq}} = \sum_{i=1}^N \left( \chi_i^0 Q_i + \frac{1}{2} J_i Q_i^2 \right) + E_{\text{PME}}(\mathbf{Q}, \mathbf{R}) + E_{\text{corr}}^{\text{Gauss}}(\mathbf{Q}, \mathbf{R}) + E_{\text{corr}}^{\text{dipole}}(\mathbf{Q}, \mathbf{R})$$

1. **On-Site Energy**: Atomic electronegativity $\chi_i^0$ and chemical hardness $J_i$ from Rappe & Goddard (1991).
2. **PME Electrostatics ($E_{\text{PME}}$)**: Real-space short-range Coulomb interactions combined with reciprocal-space FFT grid summation ($\kappa = 0.438\text{ Å}^{-1}$, 6th-order B-spline).
3. **Gaussian Charge Overlap Correction ($E_{\text{corr}}^{\text{Gauss}}$)**:
   $$E_{\text{corr}}^{\text{Gauss}} = -\frac{1}{2} \sum_{i=1}^N \sum_{j \ne i}^{N_{\text{neigh}}} Q_i Q_j \frac{\text{erfc}\left(\frac{r_{ij}}{\sqrt{2(\sigma_i^2 + \sigma_j^2)}}\right)}{r_{ij}} + \sum_{i=1}^N \frac{Q_i^2}{2\sqrt{\pi}\sigma_i}$$
   where $\sigma_i$ denotes the Gaussian charge width, parameterized strictly by covalent radii $R_{\text{covalence}}$.
4. **Dipole Slab Correction ($E_{\text{corr}}^{\text{dipole}}$)**: For interfacial cells with normal axis $z$ (volume $V$ and dipole $M_z = \sum_i Q_i z_i$):
   $$E_{\text{corr}}^{\text{dipole}} = \frac{2\pi}{V} \left( M_z^2 - Q_{\text{tot}} \sum_{i=1}^N Q_i z_i^2 - Q_{\text{tot}}^2 \frac{L_z^2}{12} \right) \xrightarrow{Q_{\text{tot}} = 0} \frac{2\pi}{V} M_z^2$$

### 1.3 Constant-Potential (ConstP) Ensemble
External electrochemical potentials $\{\phi_i\}$ shift atomic electronegativities across counter-electrodes:
$$\chi_i^{\text{eff}} = \chi_i^0 + \phi_i$$
Subject to cell electroneutrality $\sum_{i=1}^N Q_i = 0$, charges $\{Q_i\}$ and Lagrange multiplier $\chi_{\text{eq}}$ minimize the grand-canonical Lagrangian:
$$\mathcal{L}(\mathbf{Q}, \chi_{\text{eq}}) = E_{\text{QEq}}(\mathbf{Q}) + \sum_{i=1}^N \phi_i Q_i - \chi_{\text{eq}} \left( \sum_{i=1}^N Q_i \right)$$
Solved dynamically at runtime using GPU-accelerated **JAX + JAXopt L-BFGS** with analytical automatic differentiation.

### 1.4 Phase Identification: Multi-Anion Chemical Exclusion & Graph Percolation
Hu et al. (2025) identified metallic atoms via a bulk coordination cutoff (CN(Li–Li) ≥ 8), which collapses in solid electrolytes due to high packing densities and short cation–cation distances in the crystal framework (e.g., Li–Li ≈ 2.11 Å in β-Li₃N).

This framework resolves phase identification via a two-tier topological filter:
1. **Multi-Anion Chemical Exclusion**: A metal atom *i* is a metallic candidate if and only if its first coordination shell contains zero anions:
   ```text
   i ∈ Candidate  ⟺  species(i) == ACTIVE_METAL  AND  Σ_{X ∈ Anions} Σ_{j ∈ X} 1[d_ij ≤ r_cut,X] = 0
   ```
   Cutoffs are calibrated against the first radial distribution function (RDF) minimum:
   * Li–N: 2.50 Å
   * Li–P: 2.80 Å
   * Li–S: 2.90 Å
   * Li–Cl: 3.05 Å
2. **Graph Percolation from Boundary Seeds**:
   Construct graph adjacency matrix **A** over all candidates with metal–metal cutoff r_cut,metal = 3.50 Å:
   ```text
   A_ij = 1  if i, j ∈ Candidate AND d_ij ≤ r_cut,metal
   A_ij = 0  otherwise
   ```
   Electrode potential φ_bottom or φ_upper is propagated strictly to connected clusters originating from mechanical boundary seed planes. Unconnected clusters are classified as electronically isolated **Dead Metal** (φ = 0). Any cluster spanning both seed planes triggers an immediate **Physical Short-Circuit** trip.

---

## 2. Directory Layout

```
/scratch/tmp/hdang/projects/Li_Li3N/
├── active_learning/                    # AL orchestration & compute scripts
│   ├── al_config.py                    # Central configuration module
│   ├── run_al.py                       # Master Slurm pipeline controller
│   ├── check_v1_live.sh                # Live diagnostic monitor for running AL rounds
│   ├── slurm/                          # Batch execution wrappers
│   │   ├── cluster_config.sh           # HPC modules & Conda activation
│   │   ├── cpu_stage.slurm             # Generic CPU stage runner (express/normal/long)
│   │   ├── dft_dispatch.slurm          # Automated DFT queue manager
│   │   ├── run_dft.sh                  # VASP MPI runner (srun vasp_std)
│   │   ├── sample_v0.slurm             # Pretrained MACE v0 baseline sampler
│   │   ├── train_models.slurm          # MACE Full & Short training (Array 0-1)
│   │   ├── train_committee.slurm       # Committee ensemble training (Array 0-2)
│   │   ├── fresh_probe.slurm           # Production ConstP MD probe (16h GPU)
│   │   ├── probe_scoring.slurm         # Probe uncertainty scoring
│   │   ├── evaluation.slurm            # Independent test & holdout evaluation
│   │   └── convergence_report.slurm    # Scientific decision report generator
│   ├── stages/                         # Pure Python execution modules
│   │   ├── build_qeq_residual.py       # JAX QEq solver & residual dataset builder
│   │   ├── run_md.py                   # MACE+QEq NPT/NVT MD engine (ConstP/ConstQ)
│   │   ├── select_candidates.py        # Active learning frame selection
│   │   ├── prepare_dft.py              # DFT POSCAR input generator
│   │   ├── dispatch_dft.py             # Round-robin DFT submission & watchdog
│   │   ├── collect_dft.py              # vasprun.xml extractor & validator
│   │   ├── split_al.py                 # Train / Holdout dataset splitter
│   │   ├── merge_dataset.py            # Cumulative generation merger
│   │   ├── evaluate_models.py          # Metric benchmarking against DFT
│   │   ├── run_fresh_probe.py          # Fresh probe runner wrapper
│   │   ├── score_fresh_probe.py        # Uncertainty evaluator on fresh probe
│   │   └── convergence_report.py       # Scientific convergence decision gate
│   ├── template/                       # VASP input templates per chemistry
│   │   ├── Li_Li3N/                    # INCAR, KPOINTS, POTCAR for interface
│   │   ├── Li/                         # INCAR, KPOINTS, POTCAR for bulk Li
│   │   └── Li3N/                       # INCAR, KPOINTS, POTCAR for bulk Li3N
│   ├── state/                          # JSON persistent tracking records
│   │   ├── round_v*.json               # Stage progress & completion records
│   │   ├── jobs_v*.json                # Slurm Job IDs & dependency states
│   │   └── dft_v*_jobs.json            # Individual VASP job tracking
│   └── logs/                           # Standard out / err per round and stage
├── data/                               # Canonical dataset repository
│   ├── train_v*_dft.extxyz             # Accumulated DFT total energy/forces
│   ├── train_v*_short.extxyz           # Accumulated MACE short-range targets
│   ├── valid_dft.extxyz                # Validation DFT reference set
│   ├── valid_short.extxyz              # Validation short-range reference set
│   ├── test_dft.extxyz                 # Static independent test DFT set (100 frames)
│   └── test_short.extxyz               # Static independent test short-range set
├── structures/                         # Reference crystallographic inputs (.vasp)
├── trained_models/                     # Model weights & training logs
│   ├── train_full_v*/                  # MACE Full models (Total energy/forces)
│   ├── train_short_v*/                 # MACE Short models (Short-range residual)
│   └── committee_short_v*/             # MACE Committee ensemble (seeds 2-4)
├── md_validation/                      # Trajectories and evaluation reports
│   └── active_learning_v*/             # Generation workspaces
└── DFT/                                # Raw VASP calculation workspaces
    └── v*/cal/                         # Pure calculation subdirectories
```

---

## 3. Active Learning Workflow

```
[ v(k-1) Fresh Probe MD ]  (Inherited trajectory: 2501 frames, 300 K, 50 ps, ConstP)
           │
           ▼
[ committee_scoring ]      --> Evaluate committee force variance: sigma_F
           │
           ▼
[ candidate_selection ]    --> Select 100 frames: 65 High-Uncertainty + 15 Coverage + 20 Holdout
           │
           ▼
[ prepare_dft ]            --> Generate VASP calculation directories (POSCAR, INCAR, KPOINTS, POTCAR)
           │
           ▼
[ dft (al_dft_mgr) ]       --> Round-robin multi-partition Slurm submission with watchdog monitoring
           │
           ▼
[ collect_dft ]            --> Verify electronic SCF convergence (vasprun.xml: </modeling>)
           │
           ▼
[ qeq_residual ]           --> Compute E_QEq, F_QEq via JAX PME; generate E_Short, F_Short targets
           │
           ▼
[ split_al ]               --> Split current batch: 80 Train (Enrichment) + 20 Holdout
           │
           ▼
[ merge_dataset ]          --> Merge v(k-1) training data with current enrichment data
           │
           ▼
[ train_models ]           --> Retrain MACE Full and MACE Short models (120 epochs, seed 20261004)
[ train_committee ]        --> Retrain 3 MACE Committee models (120 epochs, seeds 20261005..20261007)
           │
           ▼
[ evaluation ]             --> Benchmark force/energy RMSE on independent test & holdouts
           │
           ▼
[ fresh_probe ]            --> Execute 50 ps ConstP MD at 300 K using updated MACE Short v(k)
           │
           ▼
[ probe_scoring ]          --> Score uncertainty distribution on fresh probe trajectory
           │
           ▼
[ convergence_report ]     --> Quantitative Decision Gate: CONTINUE_AL / CONVERGED / REVIEW
```

---

## 4. Execution & Operational Procedures

All active learning rounds are operated via `active_learning/run_al.py`.

### 4.1 Checking Pipeline Status
To inspect stage completion and running Slurm job IDs for the active generation:
```bash
cd /scratch/tmp/hdang/projects/Li_Li3N/active_learning
python3 run_al.py --status
```

### 4.2 Launching an Active Learning Round
To automatically submit or resume an active learning round:
```bash
cd /scratch/tmp/hdang/projects/Li_Li3N/active_learning
python3 run_al.py --round 1 --run-round
```
* The pipeline checks existing artifacts. Stages that have already produced valid output are automatically marked `DONE` and bypassed.
* If pre-DFT stages are running, execution pauses after submitting the candidate selection chain.
* Once candidates are selected, re-invoking `--run-round` submits the `dft` dispatcher (`al_dft_mgr`), which binds the entire downstream post-DFT pipeline using Slurm dependencies (`--dependency=afterok:<jobid>`).

### 4.3 Real-Time VASP Watchdog Monitoring
Run the diagnostic script to monitor VASP calculations and downstream job health:
```bash
bash /scratch/tmp/hdang/projects/Li_Li3N/active_learning/check_v1_live.sh
```

---

## 5. HPC Runtime & Scheduling Architecture

Configured for high-throughput execution across heterogeneous nodes:

| Pipeline Stage | Executable / Module | Resource Allocation | Partition Route | Walltime |
| :--- | :--- | :--- | :--- | :--- |
| **DFT Manager** | `stages/dispatch_dft.py` | 1 Node, 2 CPU, 8 GB RAM | `normal,long` | 24:00:00 |
| **VASP Calculations** | `slurm/run_dft.sh` (`vasp_std`) | 1 Node, 24 CPU, 48 GB RAM | `normal,long,requeue,requeue-zen,zen4,himszen4,hims,requeue-gpu,q0heuer` | 12:00:00 |
| **CPU Stages** | `select_candidates`, `collect_dft`, `split_al`, `merge_dataset` | 1 Node, 4 CPU, 8 GB RAM | `express,normal,long` | 01:00:00 |
| **QEq Residual** | `stages/build_qeq_residual.py` | 1 Node, 1 GPU, 8 CPU, 32 GB RAM | `himsa40,requeue-gpu` | 08:00:00 |
| **MACE Model Training** | `mace_run_train` (Array 0-1) | 1 Node, 1 GPU, 8 CPU, 32 GB RAM | `himsa40,requeue-gpu` | 08:00:00 |
| **MACE Committee** | `mace_run_train` (Array 0-2) | 1 Node, 1 GPU, 8 CPU, 32 GB RAM | `himsa40,requeue-gpu` | 08:00:00 |
| **Evaluation** | `stages/evaluate_round.py` | 1 Node, 1 GPU, 8 CPU, 32 GB RAM | `gpuexpress,himsa40,requeue-gpu` | 01:30:00 |
| **Fresh Probe MD** | `stages/run_fresh_probe.py` | 1 Node, 1 GPU, 18 CPU, 32 GB RAM | `gpu4090,requeue-gpu,gpuh200` | 16:00:00 |
| **Probe Scoring** | `stages/score_fresh_probe.py` | 1 Node, 1 GPU, 4 CPU, 32 GB RAM | `gpuexpress,himsa40,requeue-gpu` | 01:00:00 |
| **Convergence Gate** | `stages/convergence_report.py`| 1 Node, 2 CPU, 8 GB RAM | `express,normal,long` | 00:30:00 |

### 5.1 Fault Tolerance & Automatic Recovery
The automated DFT manager (`stages/dispatch_dft.py`) executes a 300-second watchdog cycle:
1. **Corrupted File Purging**: Upon VASP timeout or crash, all partial runtime outputs (`WAVECAR`, `CHGCAR`, `vasprun.xml`) are unlinked to eliminate restart read errors.
2. **Abnormal Termination Detection**: Calculations with `vasp.err` size $> 0$ or non-zero Slurm exit codes are immediately detected, cleared, and resubmitted with exponential backoff up to 5 retries.
3. **Queue Re-routing**: Calculations stuck in `PENDING` state for $> 30\text{ minutes}$ due to partition saturation are migrated dynamically using `scontrol update Partition=<next_partition>`.

---

## 6. Scientific Convergence Criteria

The generation recommendation in `convergence_report.json` is evaluated using strict quantitative gates:

1. **Uncertainty Target Gate**:
   $$\text{Fraction}\left(\sigma_{F,\text{p95}} \ge 0.100\text{ eV/Å}\right) \le 1.0％$$
2. **Uncertainty Hard Ceiling Gate**:
   $$\text{Fraction}\left(\sigma_{F,\text{p95}} \ge 0.150\text{ eV/Å}\right) == 0.0％$$
3. **Independent Test Generalization**:
   $$\Delta \text{RMSE}_F(\text{Test}) \le 10.0％ \quad \text{relative to parent round}$$
4. **Historical Holdout Preservation**:
   $$\Delta \text{RMSE}_F(\text{Holdout}_j) \le 20.0％ \quad \forall j \le k$$
5. **Enrichment Learning Condition**:
   $$\text{RMSE}_F^{(k)}(\text{Enrichment}_k) < \text{RMSE}_F^{(k-1)}(\text{Enrichment}_k)$$

* If condition (1) or (2) fails: **`CONTINUE_AL`** (active learning automatically schedules the next round).
* If all conditions (1)–(5) pass: **`CONVERGED`** (potential energy surface is production-ready).
* If regression conditions (3) or (4) fail: **`REVIEW`** (pipeline halts for human scientific assessment).
