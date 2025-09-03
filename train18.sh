# train imagenet script 
export CUDA_VISIBLE_DEVICES=4,5,6,7 # 0,1,2,3 # 
python -m torch.distributed.launch --nproc_per_node=4 --master_port 1666 --use_env train.py --lr_scheduler cos --cos_lr_T 320 --model spiking_resnet18 \
-b 128 --output_dir ./logs --tb --print-freq 1500 --amp --cache-dataset --T 4 --lr 0.1 --epochs 320 --step_size 20 \
--data_path yourpath --tet -j 8 \

