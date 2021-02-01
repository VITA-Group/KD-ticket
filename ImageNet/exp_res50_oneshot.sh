# ResNet-50 on ImageNet

# ################## Baseline ##################
python -u baseline.py --depth 50 --save_dir  ckpt --apex 1 --gpu 0 1 2 3


# ################## One-shot Pruning ##################
# 1 - 0.8^2 = 0.36
python prune.py --depth 50 --percent 0.36 --resume ckpt/90_checkpoint.pth.tar --save_dir ckpt_oneshot_36 --apex 1 --gpu 0 1 2 3
# 1 - 0.8^4 = 0.5904
python prune.py --depth 50 --percent 0.5904 --resume ckpt/90_checkpoint.pth.tar --save_dir ckpt_oneshot_59 --apex 1 --gpu 0 1 2 3
# 1 - 0.8^6 = 0.73785
python prune.py --depth 50 --percent 0.73785 --resume ckpt/90_checkpoint.pth.tar --save_dir ckpt_oneshot_74 --apex 1 --gpu 0 1 2 3
# 1 - 0.8^8 = 0.83222
python prune.py --depth 50 --percent 0.83222 --resume ckpt/90_checkpoint.pth.tar --save_dir ckpt_oneshot_83 --apex 1 --gpu 0 1 2 3
# 1 - 0.8^10 = 0.89262
python prune.py --depth 50 --percent 0.89262 --resume ckpt/90_checkpoint.pth.tar --save_dir ckpt_oneshot_89 --apex 1 --gpu 0 1 2 3


# ################## Rewind  6 epochs ##################
python -u lottery_ticket.py --depth 50 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/6_checkpoint.pth.tar --save_dir ckpt_rw_p89_e6 --gpu-id 0 1 2 3
python -u lottery_ticket.py --depth 50 --resume ckpt_oneshot_83/pruned.pth.tar --model ckpt/6_checkpoint.pth.tar --save_dir ckpt_rw_p83_e6 --gpu-id 0 1 2 3
python -u lottery_ticket.py --depth 50 --resume ckpt_oneshot_74/pruned.pth.tar --model ckpt/6_checkpoint.pth.tar --save_dir ckpt_rw_p74_e6 --gpu-id 0 1 2 3
python -u lottery_ticket.py --depth 50 --resume ckpt_oneshot_59/pruned.pth.tar --model ckpt/6_checkpoint.pth.tar --save_dir ckpt_rw_p59_e6 --gpu-id 0 1 2 3
python -u lottery_ticket.py --depth 50 --resume ckpt_oneshot_36/pruned.pth.tar --model ckpt/6_checkpoint.pth.tar --save_dir ckpt_rw_p36_e6 --gpu-id 0 1 2 3


# ################## KD + Rewind 6 epochs ##################
python -u kd_ticket.py --depth 50 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/6_checkpoint.pth.tar --teacher ckpt/90_checkpoint.pth.tar --save_dir ckpt_kdrw_89 --eskd 60 --gpu-id 0 1 2 3
python -u kd_ticket.py --depth 50 --resume ckpt_oneshot_83/pruned.pth.tar --model ckpt/6_checkpoint.pth.tar --teacher ckpt/90_checkpoint.pth.tar --save_dir ckpt_kdrw_83 --eskd 60 --gpu-id 0 1 2 3
python -u kd_ticket.py --depth 50 --resume ckpt_oneshot_74/pruned.pth.tar --model ckpt/6_checkpoint.pth.tar --teacher ckpt/90_checkpoint.pth.tar --save_dir ckpt_kdrw_74 --eskd 60 --gpu-id 0 1 2 3
python -u kd_ticket.py --depth 50 --resume ckpt_oneshot_59/pruned.pth.tar --model ckpt/6_checkpoint.pth.tar --teacher ckpt/90_checkpoint.pth.tar --save_dir ckpt_kdrw_59 --eskd 60 --gpu-id 0 1 2 3
python -u kd_ticket.py --depth 50 --resume ckpt_oneshot_36/pruned.pth.tar --model ckpt/6_checkpoint.pth.tar --teacher ckpt/90_checkpoint.pth.tar --save_dir ckpt_kdrw_36 --eskd 60 --gpu-id 0 1 2 3


# Lottery Ticket ##############################
python lottery_ticket.py --depth 50 --resume ckpt_oneshot_36/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_36 --gpu-id 0 1 2 3
python lottery_ticket.py --depth 50 --resume ckpt_oneshot_59/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_59 --gpu-id 0 1 2 3
python lottery_ticket.py --depth 50 --resume ckpt_oneshot_74/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_74 --gpu-id 0 1 2 3
python lottery_ticket.py --depth 50 --resume ckpt_oneshot_83/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_83 --gpu-id 0 1 2 3
python lottery_ticket.py --depth 50 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_89 --gpu-id 0 1 2 3


