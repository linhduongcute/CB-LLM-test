set -x

safe_dataset="${1//\//_}"
safe_model="${2//\//_}"

python get_concept_labels.py --dataset="${1}"

python train_CBL.py --automatic_concept_correction --dataset="${1}" --backbone="${2}" --tune_cbl_only

python train_FL.py --cbl_path="mpnet_acs/${safe_dataset}/${safe_model}_cbm/cbl_no_backbone_acc.pt" --dataset="${1}" --backbone="${2}"

set +x