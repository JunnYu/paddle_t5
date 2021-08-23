
accelerate launch eval_huggingface.py --model_name_or_path ../step-25000-aistudio-20.35 --length_penalty 1.0 --num_beams 4 --per_device_eval_batch_size 16 --max_source_length 1024 --max_target_length 512 --evaluate_file ../caches/cnndailymail/cnn_dailymail_dev.json 
