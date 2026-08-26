#!/bin/bash -l
#SBATCH -J rescore_phi_equil
#SBATCH -o logs/rescore_phi_equil_%j.out
#SBATCH -e logs/rescore_phi_equil_%j.err
#SBATCH -A durham
#SBATCH -p dine2
#SBATCH -t 00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# Pure re-analysis of already-saved pilot chains -- no sampling, no TF.
# Separates "slow but stationary" from "not equilibrated" (see the script
# docstring); decides whether the coverage ensemble is feasible on chains
# the project already has.

export TMPDIR="/tmp/rescore_phi_equil_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export PYTHONPATH=diffcmb

cd /cosma/apps/durham/dc-hick2/diffcmb

.venv/bin/python scripts/rescore_phi_equilibration.py \
    results/analysis/pilot_coverage_lmax64_hmc_ordfix_n3300.npz \
    results/analysis/pilot_coverage_lmax64_block_cl4_n3300.npz \
    results/analysis/pilot_coverage_lmax128_block_n3300.npz

rm -rf "$TMPDIR"
