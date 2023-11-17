export MODEL_DIR="/home/hice1/mnigam9/scratch/cache/zeroscope_v2_576w"
export OUTPUT_DIR="/home/hice1/mnigam9/scratch/cache/zeroscope_cd_distill"
export DATASET="/home/hice1/mnigam9/scratch/cache/webvid/val_dataset/{00000..00004}.tar"
python train_lcm_distill_sd_wds_video.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=256 \
    --learning_rate=1e-6 --loss_type="huber" --ema_decay=0.95 --adam_weight_decay=0.0 \
    --max_train_steps=1000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url=$DATASET \
    --validation_steps=200 \
    --checkpointing_steps=200 --checkpoints_total_limit=10 \
    --train_batch_size=2 \
    --per_video_frames=8 \
    --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest