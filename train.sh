python train.py \
    --dataset-name CIFAR100 \
    --teacher-model google/vit-large-patch16-224-in21k \
    --log-folder embed-KD-model \
    --ckpt-save-name embed-KD-model.pt \
    --train-type embed-KD
