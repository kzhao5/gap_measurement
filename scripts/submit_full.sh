#!/bin/bash
# Submit the full experiment: generation (16x 1-GPU array tasks), then
# recompute (32) + controls (32) gated on generation completing OK.
set -e
cd /home/kzhao2/gap_measurement

GEN=$(sbatch --parsable slurm/gen.sbatch)
echo "gen array:      $GEN (0-15: moe shards 0-7, dense shards 0-7)"

RC=$(sbatch --parsable --dependency=afterok:$GEN slurm/recompute.sbatch)
echo "recompute:      $RC (afterok:$GEN; arch x {bf16,fp32} x 8 shards)"

CT=$(sbatch --parsable --dependency=afterok:$GEN slurm/controls.sbatch)
echo "controls:       $CT (afterok:$GEN; C2 rerun-pair + C1 prefill, 5%)"

echo
echo "watch: squeue -u \$USER -n gap-gen,gap-recomp,gap-ctrl"
echo "after all finish, run on a login node:"
echo "  .venv/bin/python analysis/build_dataset.py moe && .venv/bin/python analysis/build_dataset.py dense"
echo "  .venv/bin/python analysis/stats_report.py moe && .venv/bin/python analysis/stats_report.py dense"
echo "  .venv/bin/python analysis/figures.py moe && .venv/bin/python analysis/figures.py dense"
echo "  .venv/bin/python analysis/verdict.py"
