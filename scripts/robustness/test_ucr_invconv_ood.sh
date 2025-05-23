model_name=InvConv
epochs=500 
patience=100 
batch_size=32 
lradj=cosanneal
lr=0.01

python -u run.py \
  --task_name classification \
  --is_training 0 \
  --root_path ./dataset/UCRArchive_2018/HandOutlines \
  --model_id HandOutlines \
  --model $model_name \
  --data UCR \
  --e_layers 3 \
  --batch_size $batch_size \
  --d_model 128 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj \
  --no_normalize \
  --augmentation None \
  --inv_ablation 1 \
  --ood_test 1

python -u run.py \
  --task_name classification \
  --is_training 0 \
  --root_path ./dataset/UCRArchive_2018/HandOutlines \
  --model_id HandOutlines \
  --model $model_name \
  --data UCR \
  --e_layers 3 \
  --batch_size $batch_size \
  --d_model 128 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj \
  --no_normalize \
  --augmentation None \
  --inv_ablation 2 \
  --ood_test 1

python -u run.py \
  --task_name classification \
  --is_training 0 \
  --root_path ./dataset/UCRArchive_2018/HandOutlines \
  --model_id HandOutlines \
  --model $model_name \
  --data UCR \
  --e_layers 3 \
  --batch_size $batch_size \
  --d_model 128 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj \
  --no_normalize \
  --augmentation None \
  --inv_ablation 3 \
  --ood_test 1