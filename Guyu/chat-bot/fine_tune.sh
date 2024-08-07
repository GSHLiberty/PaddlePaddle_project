CUDA_VISIBLE_DEVICES=0 \
python3 -u train.py --embed_dim 768 \
                      --ff_embed_dim 3072 \
                      --num_heads 12 \
                      --layers 12 \
                      --dropout 0.2 \
                      --train_data ./data/train.txt \
                      --dev_data ./data/dev.txt \
                      --vocab ../model/12L_10G.vocab.txt \
                      --min_occur_cnt 0 \
                      --batch_size 16 \
                      --warmup_steps 1 \
                      --lr 1e-2 \
                      --weight_decay 0 \
                      --smoothing 0.1 \
                      --max_len_x 64 \
                      --min_len_x 1 \
                      --max_len_y 64 \
                      --min_len_y 1 \
                      --world_size 1 \
                      --gpus 1 \
                      --start_rank 0 \
                      --MASTER_ADDR localhost \
                      --MASTER_PORT 28512 \
                      --print_every 100 \
                      --save_every 10000 \
                      --epoch 100 \
                      --save_dir ckpt \
                      --backend nccl \
                      --start_from ../model/12L_10G.ckpt
