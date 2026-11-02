#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec uaml-ab benchmark \
  --configs "$project_root/configs/adams_cdr1h.yaml" "$project_root/configs/adams_cdr3h.yaml" \
  "$@"
