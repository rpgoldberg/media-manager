#!/usr/bin/env bash
# mask-batch.sh -- run the MASK-BATCH-1 driver (python -m app.cli.mask_batch run) in the
# media-manager CUDA worker image on this box's GPU.
#
#   scripts/mask-batch.sh STATE_DIR [run options...]
#
# STATE_DIR (created if missing) is mounted at /state: put the work lists there and pass them as
# --worklist /state/<file>; the progress file is /state/progress.jsonl and the per-chunk JSON lines
# are appended to /state/batch.log as well as printed. Everything after STATE_DIR goes to `run`
# (see `python -m app.cli.mask_batch run --help`).
#
# Configuration comes from THIS shell's environment and is passed by NAME only (docker -e NAME), so
# no secret value is ever on a command line:
#   MASK_BATCH_IMAGE      the worker image (required), e.g. ghcr.io/figurecollecting/media-manager-worker@sha256:...
#   DATABASE_URL          the media database, e.g. through the operator's port-forward on 127.0.0.1
#   PGPASSWORD PGSSLMODE  libpq settings the URL leaves out
#   PGSSLROOTCERT         a CA file on this box: mounted read-only and pointed at inside the container
#   S3_ENDPOINT_URL S3_REGION S3_BUCKET S3_ACCESS_KEY S3_SECRET_KEY S3_SECURE   the media bucket
#   RAW_S3_ENDPOINT_URL RAW_S3_REGION RAW_S3_BUCKET RAW_S3_ACCESS_KEY RAW_S3_SECRET_KEY   the raw
#                         bucket, read only to the batch (the driver refuses any write to it)
#   MASK_BATCH_RESIDENTIAL_IP   the box's residential address (the VPN check)
#   MASK_STORE_GRANTS MASK_STORE_REVOKED   the deployment's mask policy, as media-manager reads it
#   ENVIRONMENT           required by media-manager's settings (no default): production for the deployed media
#   DB_POOL_RECYCLE      the pool's recycle seconds when set (-1 through a port-forward: a recycle reconnects
#                         through the tunnel); media-manager's settings read it, default 1800
#   MASK_GPU_MEM_BUDGET_MB   the TOTAL GPU memory (MiB, default 8800, minimum 6144) split across the model
#                         sessions: BiRefNet 2/3 of it, each IS-Net 1/6
#   MASK_BATCH_NAME       the container name (default mask-batch): one batch at a time; stop it
#                         with `docker stop -t 120 mask-batch` (it finishes the images past the GPU;
#                         the others in flight are not recorded and run again first)
#   MASK_BATCH_GPUS       docker --gpus value (default all; "none" runs without a GPU)
#   MASK_BATCH_MEMORY     docker --memory. Default 8g on the GPU (one image at a time the run peaked
#                         near 2.3 GB of RAM; the pipeline adds about 1.2-1.4 GB whatever --writers:
#                         --decode-slots copies decoded for the model at most (three; an original
#                         over 4000x3000 takes a slot per 4000x3000), two originals read ahead, and
#                         two malloc arenas (MALLOC_ARENA_MAX=2) rather than one kept per worker:
#                         simulated at 4000x3000, 0.9 GB one at a time, 2.1 GB at 4 writers, 2.2 GB
#                         at 12; rolling (MASK-PERF-3), 2.6 GB at 16 writers with 3 slots and 2 read
#                         ahead, 2.9 GB with 4 and 4); 10g when the model may run on the CPU
#                         (--device auto or cpu, or MASK_BATCH_GPUS=none): BiRefNet + two IS-Net
#                         sessions on the CPU are OOM-killed at 8g (exit 137 on the same image every
#                         rerun). 10g measured.
#   MASK_BATCH_APP_DIR    TESTS ONLY: mount this app/ over the image's /app/app
set -euo pipefail

if [ $# -lt 1 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,/^set -euo/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'
  exit 2
fi
state="$1"; shift
: "${MASK_BATCH_IMAGE:?set MASK_BATCH_IMAGE to the worker image}"
mkdir -p "$state"
state="$(cd "$state" && pwd)"

gpus="${MASK_BATCH_GPUS:-all}"
memory=8g
[ "$gpus" = "none" ] && memory=10g
prev=""
for arg in "$@"; do
  case "$prev $arg" in
    "--device auto" | "--device cpu" | *" --device=auto" | *" --device=cpu") memory=10g ;;
  esac
  prev="$arg"
done

args=(run --rm --init --name "${MASK_BATCH_NAME:-mask-batch}" --network host
      --memory "${MASK_BATCH_MEMORY:-$memory}" --user "$(id -u):$(id -g)"
      -v "$state:/state" -w /state -e PYTHONPATH=/app -e MALLOC_ARENA_MAX=2)
[ "$gpus" = "none" ] || args+=(--gpus "$gpus")
for name in DATABASE_URL PGPASSWORD PGSSLMODE \
            S3_ENDPOINT_URL S3_REGION S3_BUCKET S3_ACCESS_KEY S3_SECRET_KEY S3_SECURE \
            RAW_S3_ENDPOINT_URL RAW_S3_REGION RAW_S3_BUCKET RAW_S3_ACCESS_KEY RAW_S3_SECRET_KEY \
            MASK_BATCH_RESIDENTIAL_IP MASK_GPU_MEM_BUDGET_MB MASK_STORE_GRANTS MASK_STORE_REVOKED DB_POOL_RECYCLE ENVIRONMENT; do
  [ -n "${!name+x}" ] && args+=(-e "$name")
done
if [ -n "${PGSSLROOTCERT:-}" ]; then
  args+=(-v "$PGSSLROOTCERT:/run/mask-batch/pg-ca.crt:ro" -e PGSSLROOTCERT=/run/mask-batch/pg-ca.crt)
fi
if [ -n "${MASK_BATCH_APP_DIR:-}" ]; then
  args+=(-v "$MASK_BATCH_APP_DIR:/app/app:ro")
fi
args+=("$MASK_BATCH_IMAGE" python -m app.cli.mask_batch run --progress /state/progress.jsonl "$@")

set +e
docker "${args[@]}" | tee -a "$state/batch.log"
status=${PIPESTATUS[0]}
exit "$status"
