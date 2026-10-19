#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

run_fast_gate() {
  echo "== gofmt =="
  fmt_out="$(gofmt -l .)"
  if [ -n "$fmt_out" ]; then
    echo "gofmt: files need formatting:"
    echo "$fmt_out"
    exit 1
  fi

  echo "== go vet =="
  go vet ./...

  run_contracts
  run_coverage

  echo "== golangci-lint =="
  golangci-lint run ./...

  echo "ALL GATES GREEN"
}

run_contracts() {
  echo "== test-depth contracts =="
  python3 .github/workflows/test_depth_gate.py contracts
}

run_coverage() {
  echo "== exact per-package coverage =="
  python3 .github/workflows/test_depth_gate.py coverage
}

run_fuzz() {
  fuzz_time="${FUZZ_TIME:-20s}"
  if [[ ! "$fuzz_time" =~ ^([1-9][0-9]*)(s|x)$ ]] ||
    { [ "${BASH_REMATCH[2]:-}" = s ] && [ "${BASH_REMATCH[1]:-0}" -gt 300 ]; } ||
    { [ "${BASH_REMATCH[2]:-}" = x ] && [ "${BASH_REMATCH[1]:-0}" -gt 1000000 ]; }; then
    echo "fuzz: FUZZ_TIME must be within 1s..300s or 1x..1000000x, got $fuzz_time" >&2
    exit 1
  fi

  listed="$(go test -list '^Fuzz[A-Za-z0-9_]*$' .)"
  targets="$(printf '%s\n' "$listed" | awk '/^Fuzz[A-Za-z0-9_]*$/ { print }')"
  if [ "$targets" != FuzzEmbedImageReader ]; then
    echo "fuzz: expected exactly FuzzEmbedImageReader, discovered: ${targets:-none}" >&2
    exit 1
  fi
  GOMAXPROCS=2 go test -run '^$' -fuzz '^FuzzEmbedImageReader$' -fuzztime="$fuzz_time" .
}

case "${1:-gate}" in
  gate) run_fast_gate ;;
  contracts) run_contracts ;;
  coverage) run_coverage ;;
  race)
    echo "== race =="
    go test -race -count=1 ./...
    ;;
  fuzz)
    echo "== bounded image decoder fuzz =="
    run_fuzz
    ;;
  *)
    echo "usage: $0 [gate|contracts|coverage|race|fuzz]" >&2
    exit 2
    ;;
esac
