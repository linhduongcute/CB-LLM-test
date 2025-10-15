safe_dataset="${1//\//_}"
safe_model="${3//\//_}"

# Determine concept embedding model folder
concept_base="mpnet_acs"
if [[ "$2" == *"simcse"* ]]; then
    concept_base="simcse_acs"
elif [[ "$2" == *"angle"* ]]; then
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

# Check if .npy files exist
if [[ ! -f "$train_npy" || ! -f "$val_npy" ]]; then
    python get_concept_labels.py --dataset="${1}" --concept_text_sim_model="${2}"
else
    echo "Concept label .npy files already exist. Skipping get_concept_labels.py."
fi

case "${acc_flag}-${tune_flag}" in
  true-false)  cbl_fname="cbl_no_backbone.pt" ;;
  false-true)  cbl_fname="cbl_acc.pt" ;;
  true-true)   cbl_fname="cbl_no_backbone_acc.pt" ;;
  false-false) cbl_fname="cbl.pt" ;;
esac

# Define the output CBL model path
cbl_model_path="${concept_dir}/${safe_model}_cbm/${cbl_fname}"
mkdir -p "$(dirname "$cbl_model_path")"

# Check if the CBL model already exists
if [[ ! -f "$cbl_model_path" ]]; then
    python train_CBL.py --dataset="${1}" --backbone="${3}" "${extra_args[@]}"
else
    echo "CBL model already exists at $cbl_model_path. Skipping train_CBL.py."
fi

# Continue with training final layer
python train_FL.py --cbl_path="$cbl_model_path" --dataset="${1}" --backbone="${3}"
