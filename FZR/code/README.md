# Running Command

## For NELL-ZS
### km+exp
python trainer.py --splitname new1 --pretrain_batch_size 64 --pretrain_subepoch 20 --pretrain_times 25000 --pretrain_feature_extractor --load_trained_embed --cat_exp


## For Wiki-ZS
### wiki km+exp
python trainer.py --splitname new1 --embed_dim 50 --ep_dim 100 --fc1_dim 200 --pretrain_batch_size 128 --pretrain_subepoch 30 --pretrain_times 25000 --pretrain_feature_extractor --load_trained_embed --cat_exp --dataset Wiki 
