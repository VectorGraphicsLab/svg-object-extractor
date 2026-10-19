#!/usr/bin/env bash
# Build a slim article supplement archive: selected outputs (no PNG, no smoke
# metrics), dataset JSONL, svg_annotation_library.zip, unpacked OpenRouter samples.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d)"
STAGING_NAME="svg-segmentator-slim-bundle-${STAMP}"
STAGING="/tmp/${STAGING_NAME}"
ARCHIVE_NAME="${STAGING_NAME}.tar.gz"
ARCHIVE_OUT="${REPO_ROOT}/output/${ARCHIVE_NAME}"
NO_BOXES_NAME="${STAGING_NAME}-no-extracted-boxes.tar.gz"
ARCHIVE_NO_BOXES_OUT="${REPO_ROOT}/output/${NO_BOXES_NAME}"
ZIP_OUT="${REPO_ROOT}/output/${STAGING_NAME}.zip"

OPENROUTER_DIR="$(find "${REPO_ROOT}" -maxdepth 1 -type d -name '_unpacked_openrouter_*' -print | sort | head -n1)"
if [[ -z "${OPENROUTER_DIR}" ]] || [[ ! -d "${OPENROUTER_DIR}" ]]; then
  echo "error: no _unpacked_openrouter_* directory in ${REPO_ROOT}" >&2
  exit 1
fi

rm -rf "${STAGING}"
mkdir -p "${STAGING}/output"

echo "Staging outputs (exclude *.png; gt_baseline_full also excludes *smoke*)..."
rsync -a --exclude='*.png' --exclude='*smoke*' \
  "${REPO_ROOT}/output/gt_baseline_full/" "${STAGING}/output/gt_baseline_full/"
rsync -a --exclude='*.png' \
  "${REPO_ROOT}/output/llm_bridge_full_qwen_eligible/" "${STAGING}/output/llm_bridge_full_qwen_eligible/"
rsync -a --exclude='*.png' \
  "${REPO_ROOT}/output/llm_bridge_full_gemma_eligible/" "${STAGING}/output/llm_bridge_full_gemma_eligible/"

echo "Copying dataset, library zip, OpenRouter unpack..."
cp -a "${REPO_ROOT}/dataset (1).jsonl" "${STAGING}/"
cp -a "${REPO_ROOT}/svg_annotation_library.zip" "${STAGING}/"
rsync -a "${OPENROUTER_DIR}/" "${STAGING}/$(basename "${OPENROUTER_DIR}")/"

cat > "${STAGING}/README_BUNDLE.md" << EOF
# Slim article supplement bundle (${STAMP})

## Contents

- \`output/gt_baseline_full/\` — baseline run artifacts. Raster previews **PNG omitted**; **SVG previews kept**. Files matching \`*smoke*\` omitted.
- \`output/llm_bridge_full_qwen_eligible/\`, \`output/llm_bridge_full_gemma_eligible/\` — LLM bridge runs; **PNG omitted**, **SVG kept**.
- \`dataset (1).jsonl\` — evaluation dataset.
- \`svg_annotation_library.zip\` — bundled SVG library.
- \`$(basename "${OPENROUTER_DIR}")/\` — unpacked OpenRouter sample outputs referenced in the article workflow.

## Notes

- This bundle is **not** a minimal reproducibility kit (no frozen code tree, no \`node_modules\`, no full raster previews).
- Metrics and NDJSON in \`output/\` are the recorded artifacts; omitting PNG does not change stored aggregate metrics files from the original run.

## Companion archive (no per-box SVGs)

File \`${NO_BOXES_NAME}\` in the same \`output/\` directory is **identical** to this tree except preview assets named \`extracted_box*\` (per-detection box SVGs) are **not** included. Union/GT/other SVG names are kept.
EOF

echo "Creating tar.gz ${ARCHIVE_OUT} ..."
tar -czf "${ARCHIVE_OUT}" -C "$(dirname "${STAGING}")" "$(basename "${STAGING}")"

echo "Creating zip ${ZIP_OUT} ..."
(
  cd "$(dirname "${STAGING}")" || exit 1
  rm -f "${ZIP_OUT}"
  zip -r -q "${ZIP_OUT}" "$(basename "${STAGING}")"
)

echo "Stripping extracted_box* preview files for ${ARCHIVE_NO_BOXES_OUT} ..."
BOX_N="$(find "${STAGING}" -type f -name 'extracted_box*' -print | wc -l)"
find "${STAGING}" -type f -name 'extracted_box*' -delete
echo "Removed ${BOX_N} files matching extracted_box*"

echo "Creating tar.gz ${ARCHIVE_NO_BOXES_OUT} ..."
tar -czf "${ARCHIVE_NO_BOXES_OUT}" -C "$(dirname "${STAGING}")" "$(basename "${STAGING}")"

rm -rf "${STAGING}"

ls -lh "${ARCHIVE_OUT}" "${ZIP_OUT}" "${ARCHIVE_NO_BOXES_OUT}"
echo "Done."
