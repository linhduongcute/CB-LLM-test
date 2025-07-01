set -x

safe_dataset="${1//\//_}"
safe_model="${2//\//_}"

python test_CBLLM.py --cbl_path="mpnet_acs/${safe_dataset}/${safe_model}_cbm/cbl_no_backbone_acc.pt" --sparse=$3

set +x