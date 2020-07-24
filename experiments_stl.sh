# !/bin/bash
# MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja

STL_DIR="${STL_DIR:-./datasets/stl10}"


function train_stl() {
    # 1 model_name
    python train_stl10.py \
        --batch_size 128 \
        --epochs 1000 \
        --optim sgd \
        --decay 0.0005 \
        --nesterov \
        --lr 0.1 \
        --lr_steps 300 400 600 800 \
        --lr_gamma 0.2 \
        --cuda \
        --test_epochs 50 300 400 600 800 "900|1000|20" \
        --model $1 \
        --save_model_path "" \
        --tag "" \
        --data_dir="$STL_DIR" \

}


function train_stl_64() {
    ###### Here batch size is equal 64. The learning rate is adjusted accordingly. 
    ###### The final result is very close train_stl/
    ###### Use this function if the model does not fit your GPU memory

    # 1 model_name
    python train_stl10.py \
        --batch_size 64 \
        --epochs 1000 \
        --optim sgd \
        --decay 0.0005 \
        --nesterov \
        --lr 0.05 \
        --lr_steps 300 400 600 800 \
        --lr_gamma 0.2 \
        --cuda \
        --test_epochs 50 300 400 600 800 "900|1000|20" \
        --model $1 \
        --save_model_path "" \
        --tag "minibatch_64" \
        --data_dir="$STL_DIR" \

}


model_list=(
    "wrn_16_8"
    "wrn_16_8_kanazawa"
    "wrn_16_8_xu"
    "wrn_16_8_ss"
    "wrn_16_8_dss"
    "wrn_16_8_ses_a"
    "wrn_16_8_ses_b"
    "wrn_16_8_ses_c"
)

for model_name in "${model_list[@]}"
do
    train_stl "$model_name" 
done
