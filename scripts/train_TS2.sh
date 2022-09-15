source activate 4dpls

# pre-training
python3 train_SemanticKitti.py -t 2 -s results/4DPLS_TS2_pretrain --pretrain --wandb

# training
python3 train_SemanticKitti.py -t 2 -s results/4DPLS_TS2 -p 4DPLS_TS2_pretrain -i 3 --wandb