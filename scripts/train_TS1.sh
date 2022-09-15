source activate 4dpls

# pre-training
python3 train_SemanticKitti.py -t 1 -s results/checkpoints/4DPLS_TS1_pretrain --pretrain --wandb

# training
python3 train_SemanticKitti.py -t 1 -s results/checkpoints/4DPLS_TS1 -p 4DPLS_TS1_pretrain -i 3 --wandb