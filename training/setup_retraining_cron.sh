#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/cc/mlops-proj27"
TRAINING_DIR="${PROJECT_DIR}/training"

RUN_SCRIPT="${TRAINING_DIR}/run_retraining_docker.sh"
LOG_DIR="${TRAINING_DIR}/logs"
LOG_FILE="${LOG_DIR}/retraining_cron.log"

LOCK_FILE="/tmp/proj27_retraining.lock"
CRON_TAG="PROJ27_RETRAINING_CRON"

# Run every minute.
# The flock lock prevents concurrent retraining jobs.
CRON_SCHEDULE="* * * * *"

echo "========== Setting up retraining cron =========="
echo "[INFO] Project directory: ${PROJECT_DIR}"
echo "[INFO] Training directory: ${TRAINING_DIR}"
echo "[INFO] Run script: ${RUN_SCRIPT}"
echo "[INFO] Log file: ${LOG_FILE}"
echo "[INFO] Lock file: ${LOCK_FILE}"

if [ ! -d "${TRAINING_DIR}" ]; then
  echo "[ERROR] Training directory does not exist: ${TRAINING_DIR}"
  exit 1
fi

if [ ! -f "${RUN_SCRIPT}" ]; then
  echo "[ERROR] Run script does not exist: ${RUN_SCRIPT}"
  exit 1
fi

mkdir -p "${TRAINING_DIR}/data"
mkdir -p "${TRAINING_DIR}/outputs"
mkdir -p "${LOG_DIR}"

chmod +x "${RUN_SCRIPT}"

if ! command -v flock >/dev/null 2>&1; then
  echo "[ERROR] flock is not installed or not available in PATH."
  echo "[ERROR] Please install util-linux or use an instance image that includes flock."
  exit 1
fi

FLOCK_BIN="$(command -v flock)"
BASH_BIN="$(command -v bash)"

# This cron command:
# - uses flock to prevent concurrent retraining
# - runs the retraining Docker script
# - appends stdout/stderr to the cron log file
CRON_CMD="${CRON_SCHEDULE} ${FLOCK_BIN} -n ${LOCK_FILE} ${BASH_BIN} ${RUN_SCRIPT} >> ${LOG_FILE} 2>&1 # ${CRON_TAG}"

TMP_CRON="$(mktemp)"

# Keep existing cron jobs, but remove previous versions of this retraining cron.
crontab -l 2>/dev/null | grep -v "${CRON_TAG}" | grep -v "${RUN_SCRIPT}" > "${TMP_CRON}" || true

echo "${CRON_CMD}" >> "${TMP_CRON}"

crontab "${TMP_CRON}"
rm -f "${TMP_CRON}"

echo "========== Retraining cron installed =========="
echo "[INFO] Current retraining cron entry:"
crontab -l | grep "${CRON_TAG}" || true

echo ""
echo "[INFO] To view cron logs:"
echo "tail -f ${LOG_FILE}"

echo ""
echo "[INFO] To disable this retraining cron later:"
echo "crontab -l | grep -v '${CRON_TAG}' | crontab -"

echo ""
echo "[INFO] Setup finished successfully."
