#!/usr/bin/env bash
# Dynamic stage: VISR-conditioned OmniAvatar. Run from tmm/:
#   bash train_dynamic_claude/run.sh {prepare|smoke|train|infer}
set -euo pipefail
PY=${PY:-env/.venv/bin/python}
TORCHRUN=${TORCHRUN:-env/.venv/bin/torchrun}
NPROC=${NPROC:-8}
INDEX=${INDEX:-data_cache/dyn_index}
VISR=${VISR:-train_static_claude/runs/visr_v41_a/best.pt}
RUN=${RUN:-train_dynamic_claude/runs/vco_v2}
INIT=${INIT:-train_dynamic_claude/runs/vco_a/step_0030000.pt}   # v2 warm start; INIT= for scratch
AVA=${AVA-"--ava-root data/ava256_rgb_4TB --ava-pack data_cache/ava_packs"}   # AVA= to disable
EXTRA=${EXTRA:-}
export OMP_NUM_THREADS=4 PYTHONPATH=$PWD:$PWD/OmniAvatar

case "${1:-}" in
  prepare)   # one process per GPU (parser head crops), ~30-60 min; then merge
    for i in $(seq 0 $((NPROC - 1))); do
      CUDA_VISIBLE_DEVICES=$i $PY -m train_dynamic_claude.prepare --out $INDEX \
        --shard $i --num-shards $NPROC --threads 16 > $INDEX.shard$i.log 2>&1 &
    done
    wait
    $PY -m train_dynamic_claude.prepare --out $INDEX --merge ;;
  smoke)     # 1 GPU: memory (batch 4 + Ava steps), speed, one validation render
    rm -rf train_dynamic_claude/runs/smoke_v2
    $PY -m train_dynamic_claude.train --index $INDEX --visr $VISR $AVA ${INIT:+--init $INIT} \
      --steps 24 --log-every 2 --val-every 24 --val-batches 4 --render-count 1 \
      --render-steps 10 --save-every 24 --out train_dynamic_claude/runs/smoke_v2 $EXTRA ;;
  train)
    if [ -n "${RESUME:-}" ]; then START="--resume $RESUME"; else START="${INIT:+--init $INIT}"; fi
    $TORCHRUN --standalone --nproc_per_node=$NPROC -m train_dynamic_claude.train \
      --index $INDEX --visr $VISR $AVA --steps ${STEPS:-20000} --warmup 300 --out $RUN \
      $START $EXTRA ;;
  probe)     # CK=... : does the generator follow the VISR geometry? (1 GPU, ~20-40 min)
    $PY -m train_dynamic_claude.probe --checkpoint ${CK:?} --visr $VISR --index $INDEX \
      --samples ${SAMPLES:-24} --out ${OUT:-$(dirname ${CK})/probe} ;;
  eval)      # CK=... : standard metrics. Generation sharded over NPROC GPUs, then scoring.
    EOUT=${EOUT:-$(dirname ${CK:?})/eval_$(basename ${CK} .pt)}
    $TORCHRUN --standalone --nproc_per_node=$NPROC -m train_dynamic_claude.evaluate generate \
      --checkpoint $CK --visr $VISR --index $INDEX --out $EOUT --per-task ${PER_TASK:-50} \
      --seconds ${SECONDS_EVAL:-3} --motion ${MOTION:-audio}
    $PY -m train_dynamic_claude.evaluate score --out $EOUT ;;
  infer)     # CK=... IMG=... AUDIO=... [YAW="0:0,50:35,100:-35"] [DRIVER=video.mp4]
    $PY -m train_dynamic_claude.infer --checkpoint ${CK:?} --visr $VISR --image ${IMG:?} \
      --audio ${AUDIO:?} --yaw "${YAW:-0:0}" ${DRIVER:+--driver $DRIVER} --out ${OUT:-out.mp4} ;;
  *) echo "usage: $0 {prepare|smoke|train|probe|eval|infer}"; exit 1 ;;
esac
