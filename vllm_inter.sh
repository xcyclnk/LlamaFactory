cd /data/xink/projects/LlamaFactory

CUDA_VISIBLE_DEVICES=0 \
/data/xink/projects/LlamaFactory/.venv-vllm/bin/python scripts/vllm_infer.py \
  --model_name_or_path /data/LLM_model/Qwen3-VL-8B-Instruct \
  --template qwen3_vl_nothink \
  --dataset videomme_event_frames \
  --dataset_dir /data/xink/projects/data/Video-MME/event_llamafac \
  --cutoff_len 32768 \
  --video_fps 1 \
  --video_maxlen 128 \
  --image_min_pixels 4096 \
  --image_max_pixels 65536 \
  --max_new_tokens 128 \
  --batch_size 32 \
  --save_name /data/xink/projects/LlamaFactory/saves/predict_base/videomme_event_qwen3-vl-8b-instruct/generated_predictions.jsonl