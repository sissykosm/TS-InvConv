model_name=InvConv
epochs=100
patience=20

batch_size=16 
lradj=cosanneal
lr=0.001

python run.py \
  --task_name classification_pt \
  --is_training 1 \
  --root_path ../dataset/fault_diagnosis/ \
  --model_id Fault-diagnosis \
  --model $model_name \
  --data class_pt \
  --e_layers 3 \
  --batch_size $batch_size \
  --d_model 256 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj \
  --tl_source 'A'

python run.py \
  --task_name classification_pt \
  --is_training 1 \
  --root_path ../dataset/fault_diagnosis/ \
  --model_id Fault-diagnosis \
  --model $model_name \
  --data class_pt \
  --e_layers 3 \
  --batch_size $batch_size \
  --d_model 256 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj \
  --tl_source 'B'

python run.py \
  --task_name classification_pt \
  --is_training 1 \
  --root_path ../dataset/fault_diagnosis/ \
  --model_id Fault-diagnosis \
  --model $model_name \
  --data class_pt \
  --e_layers 3 \
  --batch_size $batch_size \
  --d_model 256 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj \
  --tl_source 'C'

python run.py \
  --task_name classification_pt \
  --is_training 1 \
  --root_path ../dataset/fault_diagnosis/ \
  --model_id Fault-diagnosis \
  --model $model_name \
  --data class_pt \
  --e_layers 3 \
  --batch_size $batch_size \
  --d_model 256 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj \
  --tl_source 'D'
