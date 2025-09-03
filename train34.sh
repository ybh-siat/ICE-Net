# train imagenet script 
export CUDA_VISIBLE_DEVICES=4,5,6,7 # 4,5,6,7 # 
python -m torch.distributed.launch --nproc_per_node=4 --master_port 1269 --use_env train.py --lr_scheduler cos --cos_lr_T 320 --model sew_resnet34 \
-b 120 --output_dir ./logs --tb --print-freq 3000 --amp --cache-dataset --T 4 --lr 0.1 --epoch 320 \
--data_path yourpath --tet -j 8 \
