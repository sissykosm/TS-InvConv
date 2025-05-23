model_name=VanillaConv
epochs=100
patience=20
batch_size=16
lradj=none
lr=0.001

python run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/AtrialFibrillation/ \
  --model_id AtrialFibrillation \
  --model $model_name \
  --data UEA \
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
  --lradj $lradj 

python run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/ArticularyWordRecognition/ \
  --model_id ArticularyWordRecognition \
  --model $model_name \
  --data UEA \
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
  --lradj $lradj 

python run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/BasicMotions/ \
  --model_id BasicMotions \
  --model $model_name \
  --data UEA \
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
  --lradj $lradj 
  
python run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/Cricket/ \
  --model_id Cricket \
  --model $model_name \
  --data UEA \
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
  --lradj $lradj 

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/Epilepsy/ \
  --model_id Epilepsy \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 128 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj 
  

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model $model_name \
  --data UEA \
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
  --lradj $lradj 
  

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/FaceDetection/ \
  --model_id FaceDetection \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size $batch_size \
  --d_model 64 \
  --d_ff 256 \
  --top_k 3 \
  --num_kernels 4 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/FingerMovements/ \
  --model_id FingerMovements \
  --model $model_name \
  --data UEA \
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
  --lradj $lradj 


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/HandMovementDirection/ \
  --model_id HandMovementDirection \
  --model $model_name \
  --data UEA \
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
  --lradj $lradj 

python run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/Handwriting/ \
  --model_id Handwriting \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --batch_size $batch_size \
  --d_model 32 \
  --d_ff 64 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj 


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/Heartbeat/ \
  --model_id Heartbeat \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --batch_size $batch_size \
  --d_model 64 \
  --d_ff 64 \
  --top_k 1 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj 
  
# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/Multivariate_ts/InsectWingbeat/ \
#   --model_id InsectWingbeat \
#   --model $model_name \
#   --data UEA \
#   --e_layers 3 \
#   --batch_size $batch_size \
#   --d_model 32 \
#   --d_ff 32 \
#   --top_k 3 \
#   --des 'Exp' \
#   --itr 3 \
#   --learning_rate $lr \
#   --train_epochs $epochs \
#   --patience $patience \
#   --lradj $lradj 


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/JapaneseVowels/ \
  --model_id JapaneseVowels \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --batch_size $batch_size \
  --d_model 64 \
  --d_ff 64 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj 
  

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/Libras/ \
  --model_id Libras \
  --model $model_name \
  --data UEA \
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
  --lradj $lradj 
  

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/LSST/ \
  --model_id LSST \
  --model $model_name \
  --data UEA \
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
  --lradj $lradj 
  

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/MotorImagery/ \
  --model_id MotorImagery \
  --model $model_name \
  --data UEA \
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
  --lradj $lradj 
  

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/NATOPS/ \
  --model_id NATOPS \
  --model $model_name \
  --data UEA \
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
  --lradj $lradj 
  

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/PEMS-SF/ \
  --model_id PEMS-SF \
  --model $model_name \
  --data UEA \
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
  --lradj $lradj 
  

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/PenDigits/ \
  --model_id PenDigits \
  --model $model_name \
  --data UEA \
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
  --lradj $lradj 
  

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/PhonemeSpectra/ \
  --model_id PhonemeSpectra \
  --model $model_name \
  --data UEA \
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
  --lradj $lradj 
  

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/RacketSports/ \
  --model_id RacketSports \
  --model $model_name \
  --data UEA \
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
  --lradj $lradj 
  

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/SelfRegulationSCP1/ \
  --model_id SelfRegulationSCP1 \
  --model $model_name \
  --data UEA \
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
  --lradj $lradj 


python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/SelfRegulationSCP2/ \
  --model_id SelfRegulationSCP2 \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size $batch_size \
  --d_model 64 \
  --d_ff 64 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj 
  

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/SpokenArabicDigits/ \
  --model_id SpokenArabicDigits \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --batch_size $batch_size \
  --d_model 32 \
  --d_ff 32 \
  --top_k 2 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj 
  

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/StandWalkJump/ \
  --model_id StandWalkJump \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --batch_size $batch_size \
  --d_model 32 \
  --d_ff 32 \
  --top_k 2 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj 
  

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Multivariate_ts/UWaveGestureLibrary/ \
  --model_id UWaveGestureLibrary \
  --model $model_name \
  --data UEA \
  --e_layers 2 \
  --batch_size $batch_size \
  --d_model 32 \
  --d_ff 64 \
  --top_k 3 \
  --des 'Exp' \
  --itr 3 \
  --learning_rate $lr \
  --train_epochs $epochs \
  --patience $patience \
  --lradj $lradj 
  
