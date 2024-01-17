python run_clm.py \
    --model_name_or_path ../../../../CodeGen/codegen-350M-mono \
    --train_file ../../../../CodeGen/dataset/train_segments.json \
    --validation_file ../../../../CodeGen/dataset/valid_segments.json \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ../../../../CodeGen/output \
    --num_train_epochs 50