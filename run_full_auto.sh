#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOSTS=(
  "api.stlouisfed.org"
  "query1.finance.yahoo.com"
  "guce.yahoo.com"
  "api.coingecko.com"
  "api.binance.com"
  "api.openai.com"
  "api.anthropic.com"
)

MODE="${EIMAS_AUTO_MODE:-auto}" # auto|online|offline|check

if [[ $# -gt 0 ]]; then
  case "${1}" in
    --check-only)
      MODE="check"
      shift
      ;;
    --online)
      MODE="online"
      shift
      ;;
    --offline)
      MODE="offline"
      shift
      ;;
  esac
fi

echo "========================================================"
echo "   EIMAS: Auto Full Runner"
echo "========================================================"
echo "Mode: ${MODE}"
echo

DNS_OK="false"
DNS_OUTPUT="$(python - "$MODE" "${HOSTS[@]}" <<'PY'
import socket
import sys

mode = sys.argv[1]
hosts = sys.argv[2:]

all_ok = True
for host in hosts:
    try:
        socket.getaddrinfo(host, 443)
        print(f"OK   {host}")
    except Exception as exc:
        all_ok = False
        print(f"FAIL {host} :: {exc}")

print(f"DNS_ALL_OK={str(all_ok).lower()}")
PY
)"

echo "$DNS_OUTPUT"
if echo "$DNS_OUTPUT" | grep -q "DNS_ALL_OK=true"; then
  DNS_OK="true"
fi

if [[ "$MODE" == "check" ]]; then
  echo
  echo "Check-only mode complete."
  exit 0
fi

extract_output_dir() {
  local args=("$@")
  local idx=0
  while [[ $idx -lt ${#args[@]} ]]; do
    case "${args[$idx]}" in
      --output-dir)
        if [[ $((idx + 1)) -lt ${#args[@]} ]]; then
          echo "${args[$((idx + 1))]}"
          return 0
        fi
        ;;
      --output-dir=*)
        echo "${args[$idx]#--output-dir=}"
        return 0
        ;;
    esac
    idx=$((idx + 1))
  done
  echo "outputs"
}

generate_ra_report_artifacts() {
  local output_dir="$1"
  local enabled="${EIMAS_AUTO_RA_REPORT:-true}"
  enabled="$(echo "$enabled" | tr '[:upper:]' '[:lower:]')"
  if [[ "$enabled" != "1" && "$enabled" != "true" && "$enabled" != "yes" && "$enabled" != "on" ]]; then
    echo "Skipping RA-style report export (EIMAS_AUTO_RA_REPORT=${EIMAS_AUTO_RA_REPORT:-false})"
    return 0
  fi

  echo
  echo "Generating RA-style report artifacts (MD/HTML/PDF)..."
  if python scripts/generate_final_report.py --style ra --pdf --output-dir "$output_dir"; then
    echo "RA-style report artifacts generated."
  else
    echo "WARN: RA-style report artifact export failed (pipeline output remains available)."
  fi
}

is_true() {
  local raw="${1:-}"
  raw="$(echo "$raw" | tr '[:upper:]' '[:lower:]')"
  [[ "$raw" == "1" || "$raw" == "true" || "$raw" == "yes" || "$raw" == "on" ]]
}

PAPER_ARGS=()
if is_true "${EIMAS_ENABLE_AUTO_PAPER:-false}" || is_true "${EIMAS_PAPER_POLL_ONLY:-false}"; then
  PAPER_ARGS+=(--paper-auto)
fi
if is_true "${EIMAS_PAPER_POLL_ONLY:-false}"; then
  PAPER_ARGS+=(--paper-poll-only)
fi
if is_true "${EIMAS_PAPER_BACKTEST:-false}"; then
  PAPER_ARGS+=(--paper-backtest)
fi
if is_true "${EIMAS_PAPER_ENFORCE_APPROVAL:-false}"; then
  PAPER_ARGS+=(--paper-enforce-approval)
fi
if [[ -n "${EIMAS_PAPER_ACCOUNT:-}" ]]; then
  PAPER_ARGS+=(--paper-account "${EIMAS_PAPER_ACCOUNT}")
fi
if [[ -n "${EIMAS_PAPER_CAPITAL:-}" ]]; then
  PAPER_ARGS+=(--paper-capital "${EIMAS_PAPER_CAPITAL}")
fi

if [[ "$MODE" == "online" ]]; then
  RUN_MODE="online"
elif [[ "$MODE" == "offline" ]]; then
  RUN_MODE="offline"
elif [[ "$DNS_OK" == "true" ]]; then
  RUN_MODE="online"
else
  RUN_MODE="offline"
fi

echo
echo "Selected run mode: ${RUN_MODE}"
echo

REPORT_OUTPUT_DIR="$(extract_output_dir "$@")"

if [[ "$RUN_MODE" == "online" ]]; then
  echo "Running: python main.py --full ${PAPER_ARGS[*]} $*"
  python main.py --full "${PAPER_ARGS[@]}" "$@"
  generate_ra_report_artifacts "$REPORT_OUTPUT_DIR"
  exit 0
fi

echo "Running offline-safe full mode (network fail-fast + fallback)..."
EIMAS_FRED_FAIL_FAST_NETWORK=true \
EIMAS_EXTENDED_FAIL_FAST_NETWORK=true \
EIMAS_MARKET_DATA_FAIL_FAST_NETWORK=true \
EIMAS_CRYPTO_DATA_FAIL_FAST_NETWORK=true \
EIMAS_MARKET_INDICATORS_FAIL_FAST_NETWORK=true \
EIMAS_INSTITUTIONAL_FAIL_FAST_NETWORK=true \
EIMAS_DEBATE_FAIL_FAST_NETWORK=true \
EIMAS_SKIP_ENHANCED_DEBATE=true \
EIMAS_REPORT_FAIL_FAST_NETWORK=true \
EIMAS_OFFLINE_MARKET_FALLBACK_FORCE=true \
python main.py --full --cron-mode "${PAPER_ARGS[@]}" "$@"

generate_ra_report_artifacts "$REPORT_OUTPUT_DIR"
