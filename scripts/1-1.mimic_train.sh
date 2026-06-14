dataset="mimic_cxr"
annotation="your path"
base_dir="your path"

version="pretrain"
savepath="./save/$dataset/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi
# per GPU batch size
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u pretrain.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 20 \
    --savedmodel_path ${savepath} \
    --learning_rate 5e-5 \
    --gradient_clip_val 1 \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 4 \
    --max_epochs 10 \
    --limit_val_batches 0.5 \
    --val_check_interval 0.5 \
    --seed 42 \
    2>&1 |tee -a ${savepath}/log.txt
