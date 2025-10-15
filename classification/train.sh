#!/usr/bin/env bash
set -euo pipefail

# Grab required args first
dataset="${1}"
concept_text_sim_model="${2}"
backbone="${3}"

# Safe names for paths
safe_dataset="${dataset//\//_}"
safe_model="${backbone//\//_}"

# Determine concept embedding model folder
concept_base="mpnet_acs"
if [[ "$concept_text_sim_model" == *"simcse"* ]]; then
    concept_base="simcse_acs"
elif [[ "$concept_text_sim_model" == *"angle"* ]]; then
    concept_base="angle_acs"
fi

concept_dir="${concept_base}/${safe_dataset}"
train_npy="${concept_dir}/concept_labels_train.npy"
val_npy="${concept_dir}/concept_labels_val.npy"

# Collect optional flags to pass through to train_CBL.py
acc_flag=false
tune_flag=false
extra_args=()

shift 3
while (( "$#" )); do
  case "$1" in
    --automatic_concept_correction)
      acc_flag=true
      extra_args+=("--automatic_concept_correction")
      shift
      ;;
    --tune_cbl_only)
      tune_flag=true
      extra_args+=("--tune_cbl_only")
      shift
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

# Ensure concept labels
if [[ ! -f "$train_npy" || ! -f "$val_npy" ]]; then
    python get_concept_labels.py \
      --dataset="${dataset}" \
      --concept_text_sim_model="${concept_text_sim_model}"
else
    echo "Concept label .npy files already exist. Skipping get_concept_labels.py."
fi

# Filename logic
case "${acc_flag}-${tune_flag}" in
  true-false)  cbl_fname="cbl_acc.pt" ;;   # only ACC
  false-true)  cbl_fname="cbl_no_backbone.pt" ;;           # only tune
  true-true)   cbl_fname="cbl_no_backbone_acc.pt" ;; # both flags
  false-false) cbl_fname="cbl.pt" ;;               # no flags
esac

# Define output path
cbl_model_path="${concept_dir}/${safe_model}_cbm/${cbl_fname}"
mkdir -p "$(dirname "$cbl_model_path")"

# Train CBL (pass through only the flags provided)
if [[ ! -f "$cbl_model_path" ]]; then
    python train_CBL.py \
      --dataset="${dataset}" \
      --backbone="${backbone}" \
      "${extra_args[@]}"
else
    echo "CBL model already exists at $cbl_model_path. Skipping train_CBL.py."
fi

# Train final layer
python train_FL.py \
  --cbl_path="$cbl_model_path" \
  --dataset="${dataset}" \
  --backbone="${backbone}"