#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_ROOT="${ROOT_DIR}/configs"
TRAIN_SCRIPT="${ROOT_DIR}/src/train_MiT4SL.py"

DEFAULT_CONFIG_DIR="cross_cell_line"
DEFAULT_TARGET="A549"


usage() {
  cat <<'EOF'
Usage: bash scripts/run_mit4sl.sh [options]

Use protocol.yaml + target.yaml to run MiT4SL.
Default example is the A549 cross-cell-line setting.

Options:
  --config-dir DIR        Config subdirectory under configs/ (default: cross_cell_line)
  --target NAME          Target config name or shortcut (default: A549)
  --python BIN           Python executable to use (default: $PYTHON_BIN or python)
  --device VALUE         Optional runtime device override (e.g. 0, cpu, cuda:1)
  --save-model-path DIR  Optional override for --Save_model_path
  --dry-run              Only print the resolved command, do not launch training
  --list-targets         List available targets under the selected config dir
  -h, --help             Show this help message

Examples:
  bash scripts/run_mit4sl.sh
  bash scripts/run_mit4sl.sh --dry-run
  bash scripts/run_mit4sl.sh --device 0
  bash scripts/run_mit4sl.sh --config-dir cell_line_specific/random --target A549
  bash scripts/run_mit4sl.sh --config-dir recom_sl_partner --target A549_KRAS
EOF
}


require_option_value() {
  local option_name="$1"
  local option_value="${2-}"
  if [[ -z "${option_value}" || "${option_value}" == --* ]]; then
    echo "[MiT4SL] Error: ${option_name} requires a value." >&2
    usage >&2
    exit 1
  fi
}


normalize_config_dir_arg() {
  local config_dir="$1"
  case "${config_dir}" in
    specific)
      printf '%s\n' "cell_line_specific"
      ;;
    specific/*)
      printf '%s\n' "cell_line_specific/${config_dir#specific/}"
      ;;
    *)
      printf '%s\n' "${config_dir}"
      ;;
  esac
}


list_target_configs() {
  local config_dir="$1"
  local path
  shopt -s nullglob
  for path in "${config_dir}"/*.yaml; do
    [[ "$(basename "${path}")" == "protocol.yaml" ]] && continue
    printf '%s\n' "${path}"
  done
  shopt -u nullglob
}


resolve_config_dir() {
  local config_dir="$1"
  config_dir="$(normalize_config_dir_arg "${config_dir}")"
  local resolved="${CONFIG_ROOT}/${config_dir}"
  if [[ ! -d "${resolved}" ]]; then
    echo "[MiT4SL] Error: config directory not found: ${resolved}" >&2
    return 1
  fi
  printf '%s\n' "${resolved}"
}


resolve_protocol_config() {
  local config_dir="$1"
  local protocol_path="${config_dir}/protocol.yaml"
  if [[ ! -f "${protocol_path}" ]]; then
    echo "[MiT4SL] Error: protocol config not found: ${protocol_path}" >&2
    return 1
  fi
  printf '%s\n' "${protocol_path}"
}


resolve_target_config() {
  local config_dir="$1"
  local target="$2"
  local normalized_name target_stem
  local -a candidates=() exact_name_matches=() exact_stem_matches=() suffix_matches=() prefix_matches=() fuzzy_matches=() matched_names=()
  local -A seen_paths=() seen_names=()
  local candidate candidate_name candidate_stem

  mapfile -t candidates < <(list_target_configs "${config_dir}")
  if [[ "${#candidates[@]}" -eq 0 ]]; then
    echo "[MiT4SL] Error: no target config files found in ${config_dir}" >&2
    return 1
  fi

  if [[ "${target}" == *.yaml ]]; then
    normalized_name="${target}"
    target_stem="${target%.yaml}"
  else
    normalized_name="${target}.yaml"
    target_stem="${target}"
  fi

  for candidate in "${candidates[@]}"; do
    candidate_name="$(basename "${candidate}")"
    candidate_stem="${candidate_name%.yaml}"

    [[ "${candidate_name}" == "${normalized_name}" ]] && exact_name_matches+=("${candidate}")
    [[ "${candidate_stem}" == "${target_stem}" ]] && exact_stem_matches+=("${candidate}")
    [[ "${candidate_stem}" == *"_${target_stem}" || "${candidate_stem}" == *"_to_${target_stem}" ]] && suffix_matches+=("${candidate}")
    [[ "${candidate_stem}" == "${target_stem}_"* ]] && prefix_matches+=("${candidate}")
  done

  if [[ "${#exact_name_matches[@]}" -eq 1 ]]; then
    printf '%s\n' "${exact_name_matches[0]}"
    return 0
  fi

  if [[ "${#exact_stem_matches[@]}" -eq 1 ]]; then
    printf '%s\n' "${exact_stem_matches[0]}"
    return 0
  fi

  for candidate in "${suffix_matches[@]}" "${prefix_matches[@]}"; do
    [[ -z "${candidate}" ]] && continue
    if [[ -z "${seen_paths[${candidate}]+x}" ]]; then
      fuzzy_matches+=("${candidate}")
      seen_paths["${candidate}"]=1
    fi
  done

  if [[ "${#fuzzy_matches[@]}" -eq 1 ]]; then
    printf '%s\n' "${fuzzy_matches[0]}"
    return 0
  fi

  for candidate in "${exact_name_matches[@]}" "${exact_stem_matches[@]}" "${fuzzy_matches[@]}"; do
    [[ -z "${candidate}" ]] && continue
    candidate_name="$(basename "${candidate}")"
    if [[ -z "${seen_names[${candidate_name}]+x}" ]]; then
      matched_names+=("${candidate_name}")
      seen_names["${candidate_name}"]=1
    fi
  done

  if [[ "${#matched_names[@]}" -gt 0 ]]; then
    printf '[MiT4SL] Error: target %q is ambiguous in %s. Matching configs: ' "${target}" "${config_dir}" >&2
    local index
    for index in "${!matched_names[@]}"; do
      if [[ "${index}" -gt 0 ]]; then
        printf ', ' >&2
      fi
      printf '%s' "${matched_names[${index}]}" >&2
    done
    printf '\n' >&2
    return 1
  fi

  echo "[MiT4SL] Error: target config '${target}' was not found in ${config_dir}" >&2
  echo "[MiT4SL] Available targets:" >&2
  for candidate in "${candidates[@]}"; do
    echo "  - $(basename "${candidate}" .yaml)" >&2
  done
  return 1
}


print_command() {
  local arg
  printf '[MiT4SL] Command:'
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
}


CONFIG_DIR="${DEFAULT_CONFIG_DIR}"
TARGET="${DEFAULT_TARGET}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE_OVERRIDE="${DEVICE_OVERRIDE:-}"
SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-}"
DRY_RUN=0
LIST_TARGETS=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --config-dir)
      require_option_value "$1" "${2-}"
      CONFIG_DIR="$2"
      shift 2
      ;;
    --target)
      require_option_value "$1" "${2-}"
      TARGET="$2"
      shift 2
      ;;
    --python)
      require_option_value "$1" "${2-}"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --device)
      require_option_value "$1" "${2-}"
      DEVICE_OVERRIDE="$2"
      shift 2
      ;;
    --save-model-path)
      require_option_value "$1" "${2-}"
      SAVE_MODEL_PATH="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --list-targets)
      LIST_TARGETS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[MiT4SL] Error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done


CONFIG_DIR_PATH="$(resolve_config_dir "${CONFIG_DIR}")"
if [[ "${LIST_TARGETS}" -eq 1 ]]; then
  echo "[MiT4SL] Available targets in configs/${CONFIG_DIR}:"
  while IFS= read -r target_path; do
    echo "  - $(basename "${target_path}" .yaml)"
  done < <(list_target_configs "${CONFIG_DIR_PATH}" | sort)
  exit 0
fi

PROTOCOL_CONFIG="$(resolve_protocol_config "${CONFIG_DIR_PATH}")"
TARGET_CONFIG="$(resolve_target_config "${CONFIG_DIR_PATH}" "${TARGET}")"

COMMAND=(
  "${PYTHON_BIN}"
  "${TRAIN_SCRIPT}"
  --cfg "${PROTOCOL_CONFIG}"
  --cfg "${TARGET_CONFIG}"
)

if [[ -n "${DEVICE_OVERRIDE}" ]]; then
  COMMAND+=(--device "${DEVICE_OVERRIDE}")
fi

if [[ -n "${SAVE_MODEL_PATH}" ]]; then
  COMMAND+=(--Save_model_path "${SAVE_MODEL_PATH}")
fi

echo "[MiT4SL] Repo root: ${ROOT_DIR}"
echo "[MiT4SL] Protocol config: ${PROTOCOL_CONFIG#${ROOT_DIR}/}"
echo "[MiT4SL] Target config: ${TARGET_CONFIG#${ROOT_DIR}/}"
print_command "${COMMAND[@]}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

"${COMMAND[@]}"
