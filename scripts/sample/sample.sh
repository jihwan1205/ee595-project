# CKPT_PATH="/home/jihwanshin/EE595_project/runs/meanflow_20251118_140918/best_nfe4.pt"
# CKPT_PATH="/home/jihwanshin/EE595_project/runs/fm_20251118_140918/best_nfe4.pt"
CKPT_PATH="/home/jihwanshin/EE595_project/runs/meanflow_reflow_20251125_105007/checkpoint_iter_180000.pt"

cmd="python sampling.py \
    --ckpt_path $CKPT_PATH \
    --save_dir ./samples/meanflow_reflow/180000 \
    --nfe_list 1\
    ${@:3:$#}"

echo $cmd   
eval $cmd