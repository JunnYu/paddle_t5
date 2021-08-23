# ["cola","sst-2","mrpc","sts-b","qqp","mnli", "rte", "qnli"]
# 请参考 logs/GLUE/task名字/args.json，然后配置参数！
###############################################################################################################################
# MNLI
python run_glue.py \
    --model_name_or_path ../t5-base \
    --task_name mnli \
    --max_seq_length 256 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_radio 0.1 \
    --num_train_epochs 3 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --seed 42 \
    --output_dir outputs/mnli/ \
    --device gpu \
    --num_workers 2

# QNLI
python run_glue.py \
    --model_name_or_path ../t5-base \
    --task_name qnli \
    --max_seq_length 256 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_radio 0.1 \
    --num_train_epochs 3 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --seed 42 \
    --output_dir outputs/qnli/ \
    --device gpu \
    --num_workers 2
    
# QQP
# 运行训练
python run_glue.py \
    --model_name_or_path ../t5-base \
    --task_name qqp \
    --max_seq_length 256 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_radio 0.1 \
    --num_train_epochs 3 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --seed 42 \
    --output_dir outputs/qqp/ \
    --device gpu \
    --num_workers 2

############################################################################################################################################
# COLA
python run_glue.py \
    --model_name_or_path ../t5-base \
    --task_name cola \
    --max_seq_length 256 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_radio 0.1 \
    --num_train_epochs 10 \
    --logging_steps 200 \
    --save_steps 200 \
    --seed 42 \
    --output_dir outputs/cola/ \
    --device gpu \
    --num_workers 4

  
# SST2
python run_glue.py \
    --model_name_or_path ../t5-base \
    --task_name sst-2 \
    --max_seq_length 256 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_radio 0.1 \
    --num_train_epochs 4 \
    --logging_steps 400 \
    --save_steps 400 \
    --seed 42 \
    --output_dir outputs/sst-2/ \
    --device gpu \
    --num_workers 4

########################################################
# RTE
python run_glue.py \
    --model_name_or_path ../t5-base \
    --task_name rte \
    --max_seq_length 256 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_radio 0.1 \
    --num_train_epochs 10 \
    --logging_steps 100 \
    --save_steps 100 \
    --seed 42 \
    --output_dir outputs/rte/ \
    --device gpu \
    --num_workers 4
    
############################################################
# MRPC
python run_glue.py \
    --model_name_or_path ../t5-base \
    --task_name mrpc \
    --max_seq_length 256 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_radio 0.1 \
    --num_train_epochs 10 \
    --logging_steps 100 \
    --save_steps 100 \
    --seed 42 \
    --output_dir outputs/mrpc/ \
    --device gpu \
    --num_workers 4
  
############################################################
# STSB
python run_glue.py \
    --model_name_or_path ../t5-base \
    --task_name sts-b \
    --max_seq_length 256 \
    --train_batch_size 16 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_radio 0.1 \
    --num_train_epochs 10 \
    --logging_steps 100 \
    --save_steps 100 \
    --seed 42 \
    --output_dir outputs/sts-b/ \
    --device gpu \
    --num_workers 4
  
############################################################