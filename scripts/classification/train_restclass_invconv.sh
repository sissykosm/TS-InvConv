model_name=InvConv
epochs=300
patience=20

batch_size=16 
lradj=cosanneal
lr=0.001

python run.py \
  --task_name classification_pt \
  --is_training 1 \
  --root_path ./dataset/har-uci/ \
  --model_id HAR-uci \
  --model $model_name \
  --data class_pt \
  --e_layers 3 \
  --batch_size $batch_size \
  --d_model 32 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj \

python run.py \
  --task_name classification_pt \
  --is_training 1 \
  --root_path ./dataset/sleep-edf/ \
  --model_id Sleep-EDF \
  --model $model_name \
  --data class_pt \
  --e_layers 3 \
  --batch_size $batch_size \
  --d_model 32 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj \

lradj=none

python run.py \
  --task_name classification_pt \
  --is_training 1 \
  --root_path ./dataset/epilepsy/ \
  --model_id Epilepsy \
  --model $model_name \
  --data class_pt \
  --e_layers 3 \
  --batch_size $batch_size \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj \
