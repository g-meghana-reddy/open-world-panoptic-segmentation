source activate 4dpls

# pre-training
python3 train_SemanticKitti.py -t 0 -s results/4DPLS_TS0_pretrain --pretrain --wandb

# training
python3 train_SemanticKitti.py -t 0 -s results/4DPLS_TS0 -p 4DPLS_TS0_pretrain -i 3 --wandb