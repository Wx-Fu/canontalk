#!/usr/bin/env bash
# Full static-stage pipeline on one 8-GPU node. Run from the tmm/ root:
#   bash train_static_claude/run_8gpu.sh <stage>
# stages: prep_emo | prep_ava | check | smoke | stage_a | eval_a | diagnose | stage_b | eval_b | infer
# Defaults are v3 (texture slots + colour copy + pose refinement), warm-started from
# the v2 best checkpoint. v1/v2 results stay in runs/visr_a and runs/visr_v2_a.
set -euo pipefail
PY=${PY:-env/.venv/bin/python}
TORCHRUN=${TORCHRUN:-env/.venv/bin/torchrun}
NPROC=${NPROC:-8}
AVA=${AVA:-data/ava256_rgb_4TB}
CACHE=${CACHE:-data_cache}
EMO_INDEX=$CACHE/emotalk3d_frames/index.jsonl
PACK=$CACHE/ava_packs
AVA_FRAMES=${AVA_FRAMES:-256}
RUN_A=${RUN_A:-train_static_claude/runs/visr_v41_a}
RUN_B=${RUN_B:-train_static_claude/runs/visr_v41_b}
INIT=${INIT:-train_static_claude/runs/visr_v4_a/step_0015000.pt}   # INIT= (empty) to train from scratch
DATA="--ava-root $AVA --ava-pack $PACK --emo-index $EMO_INDEX"
EXTRA=${EXTRA:-}          # e.g. EXTRA="--grad-ckpt" or EXTRA="--batch 2"
export OMP_NUM_THREADS=4
best_or_latest() {        # best.pt if it exists, else the latest step checkpoint
  if [ -f "$1/best.pt" ]; then echo "$1/best.pt"
  else echo "$1/$(python3 -c "import json;print(json.load(open('$1/latest.json'))['checkpoint'])")"; fi
}

case "${1:-}" in
  prep_emo)   # CPU, ~1-2 h with 32 processes, ~45 GB (already done for v1)
    $PY -m train_static_claude.prepare_emo \
      --manifests train/manifests/train.jsonl train/manifests/val.jsonl \
      --out $CACHE/emotalk3d_frames --frames-per-clip 6 --workers 32 ;;
  prep_ava)   # CPU, resumable; grows existing packs 128 -> 256 frames/person (~+75 GB)
    $PY -m train_static_claude.prepare_ava --ava-root $AVA --out $PACK \
      --frames-per-capture $AVA_FRAMES --processes 16 --threads 32 ;;
  check)      # look at the PNGs before training
    $PY -m train_static_claude.check_data $DATA --samples 8 \
      --out train_static_claude/runs/data_check ;;
  smoke)      # 1 GPU, 50 steps: memory (mem_gb), speed, finite losses, warm-start report
    rm -rf train_static_claude/runs/smoke_v41
    $PY -m train_static_claude.train $DATA --steps 50 --log-every 5 --val-every 50 \
      --val-batches 2 --save-every 50 ${INIT:+--init $INIT} --gt-pose-warmup 5000 \
      --out train_static_claude/runs/smoke_v41 $EXTRA ;;
  stage_a)    # warm start: short warm-up, short teacher forcing, 60k steps
    if [ -n "${RESUME:-}" ]; then START="--resume $RESUME"; else START="${INIT:+--init $INIT}"; fi
    $TORCHRUN --standalone --nproc_per_node=$NPROC -m train_static_claude.train $DATA \
      --batch 4 --workers 12 --steps ${STEPS:-30000} --lr 3e-4 --lr-vit 3e-5 --warmup 1000 \
      --gt-pose-warmup 5000 --val-every 2000 --save-every 5000 --out $RUN_A \
      $START $EXTRA ;;
  eval_a)
    CK=${CK:-$(best_or_latest $RUN_A)}
    $PY -m train_static_claude.evaluate --checkpoint $CK $DATA --split val --samples 800 \
      --out $RUN_A/eval_val.json ;;
  diagnose)   # where is the quality lost? (self-view / pose / textures / fusion)
    CK=${CK:-$(best_or_latest $RUN_A)}
    $PY -m train_static_claude.diagnose --checkpoint $CK $DATA --samples ${SAMPLES:-200} \
      --out $RUN_A/diagnose ;;
  stage_b)
    CK=${CK:-$(best_or_latest $RUN_A)}
    $TORCHRUN --standalone --nproc_per_node=$NPROC -m train_static_claude.train_prior \
      --stage-a $CK $DATA --fusion-views 4 --batch 8 --workers 12 --steps 60000 \
      --out $RUN_B ${RESUME:+--resume $RESUME} ;;
  eval_b)
    CK=${CK:-$(best_or_latest $RUN_A)}; PRIOR=${PRIOR:?set PRIOR=stage-B checkpoint}
    $PY -m train_static_claude.evaluate --checkpoint $CK --prior $PRIOR --prior-samples 3 \
      $DATA --split val --samples 800 --out $RUN_B/eval_val.json ;;
  infer)
    CK=${CK:-$(best_or_latest $RUN_A)}; IMG=${IMG:?set IMG}
    $PY -m train_static_claude.infer --checkpoint $CK ${PRIOR:+--prior $PRIOR} --image $IMG \
      --out ${OUT:-orbit.png} ;;
  *) echo "usage: $0 {prep_emo|prep_ava|check|smoke|stage_a|eval_a|diagnose|stage_b|eval_b|infer}"; exit 1 ;;
esac
