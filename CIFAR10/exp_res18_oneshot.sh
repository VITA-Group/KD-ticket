# ResNet-18 (Designed for ImageNet)

# Baseline ####################################
python cifar_baseline.py --arch resnet18 --depth 18 --save_dir  ckpt/ --gpu-id 0

# One Shot Pruning #######################################
# 1 - 0.8 = 0.2
python cifar_prune.py --arch resnet18 --depth 18 --percent 0.2 --resume ckpt/182_checkpoint.pth.tar --save_dir ckpt_oneshot_20
# 1 - 0.8^2 = 0.36
python cifar_prune.py --arch resnet18 --depth 18 --percent 0.36 --resume ckpt/182_checkpoint.pth.tar --save_dir ckpt_oneshot_36
# 1 - 0.8^3 = 0.488
python cifar_prune.py --arch resnet18 --depth 18 --percent 0.488 --resume ckpt/182_checkpoint.pth.tar --save_dir ckpt_oneshot_49
# 1 - 0.8^4 = 0.5904
python cifar_prune.py --arch resnet18 --depth 18 --percent 0.5904 --resume ckpt/182_checkpoint.pth.tar --save_dir ckpt_oneshot_59
# 1 - 0.8^5 = 0.67232
python cifar_prune.py --arch resnet18 --depth 18 --percent 0.67232 --resume ckpt/182_checkpoint.pth.tar --save_dir ckpt_oneshot_67
# 1 - 0.8^6 = 0.73785
python cifar_prune.py --arch resnet18 --depth 18 --percent 0.73785 --resume ckpt/182_checkpoint.pth.tar --save_dir ckpt_oneshot_74
# 1 - 0.8^7 = 0.79028
python cifar_prune.py --arch resnet18 --depth 18 --percent 0.79028 --resume ckpt/182_checkpoint.pth.tar --save_dir ckpt_oneshot_79
# 1 - 0.8^8 = 0.83222
python cifar_prune.py --arch resnet18 --depth 18 --percent 0.83222 --resume ckpt/182_checkpoint.pth.tar --save_dir ckpt_oneshot_83
# 1 - 0.8^9 = 0.86578
python cifar_prune.py --arch resnet18 --depth 18 --percent 0.86578 --resume ckpt/182_checkpoint.pth.tar --save_dir ckpt_oneshot_87
# 1 - 0.8^10 = 0.89262
python cifar_prune.py --arch resnet18 --depth 18 --percent 0.89262 --resume ckpt/182_checkpoint.pth.tar --save_dir ckpt_oneshot_89

# 1 - 0.8^11 = 0.91410
python cifar_prune.py --arch resnet18 --depth 18 --percent 0.91410 --resume ckpt/182_checkpoint.pth.tar --save_dir ckpt_oneshot_91
# 1 - 0.8^12 = 0.93128
python cifar_prune.py --arch resnet18 --depth 18 --percent 0.93128 --resume ckpt/182_checkpoint.pth.tar --save_dir ckpt_oneshot_93
# 1 - 0.8^13 = 0.94502
python cifar_prune.py --arch resnet18 --depth 18 --percent 0.94502 --resume ckpt/182_checkpoint.pth.tar --save_dir ckpt_oneshot_94
# 1 - 0.8^14 = 0.95602
python cifar_prune.py --arch resnet18 --depth 18 --percent 0.94502 --resume ckpt/182_checkpoint.pth.tar --save_dir ckpt_oneshot_95
# 1 - 0.8^15 = 0.964816
python cifar_prune.py --arch resnet18 --depth 18 --percent 0.964816 --resume ckpt/182_checkpoint.pth.tar --save_dir ckpt_oneshot_96
# 1 - 0.8^16 = 0.97185
python cifar_prune.py --arch resnet18 --depth 18 --percent 0.97185 --resume ckpt/182_checkpoint.pth.tar --save_dir ckpt_oneshot_97


# Lottery Ticket ##############################
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_20/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_20 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_36/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_36 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_49/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_49 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_59/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_59 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_67/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_67 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_74/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_74 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_79/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_79 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_83/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_83 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_87/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_87 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_89 --gpu-id 3

python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_91/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_91 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_93/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_93 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_94/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_94 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_95/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_95 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_96/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_96 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_97/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_97 --gpu-id 3

# Reinit ######################################
python cifar_scratch_no_longer.py --arch resnet18 --depth 18 --resume ckpt_oneshot_20/pruned.pth.tar --save_dir ckpt_ri_20 --gpu-id 2
python cifar_scratch_no_longer.py --arch resnet18 --depth 18 --resume ckpt_oneshot_36/pruned.pth.tar --save_dir ckpt_ri_36 --gpu-id 2
python cifar_scratch_no_longer.py --arch resnet18 --depth 18 --resume ckpt_oneshot_49/pruned.pth.tar --save_dir ckpt_ri_49 --gpu-id 2
python cifar_scratch_no_longer.py --arch resnet18 --depth 18 --resume ckpt_oneshot_59/pruned.pth.tar --save_dir ckpt_ri_59 --gpu-id 2
python cifar_scratch_no_longer.py --arch resnet18 --depth 18 --resume ckpt_oneshot_67/pruned.pth.tar --save_dir ckpt_ri_67 --gpu-id 2
python cifar_scratch_no_longer.py --arch resnet18 --depth 18 --resume ckpt_oneshot_74/pruned.pth.tar --save_dir ckpt_ri_74 --gpu-id 2
python cifar_scratch_no_longer.py --arch resnet18 --depth 18 --resume ckpt_oneshot_79/pruned.pth.tar --save_dir ckpt_ri_79 --gpu-id 2
python cifar_scratch_no_longer.py --arch resnet18 --depth 18 --resume ckpt_oneshot_83/pruned.pth.tar --save_dir ckpt_ri_83 --gpu-id 2
python cifar_scratch_no_longer.py --arch resnet18 --depth 18 --resume ckpt_oneshot_87/pruned.pth.tar --save_dir ckpt_ri_87 --gpu-id 2
python cifar_scratch_no_longer.py --arch resnet18 --depth 18 --resume ckpt_oneshot_89/pruned.pth.tar --save_dir ckpt_ri_89 --gpu-id 2

python cifar_scratch_no_longer.py --arch resnet18 --depth 18 --resume ckpt_oneshot_91/pruned.pth.tar --save_dir ckpt_ri_91 --gpu-id 2
python cifar_scratch_no_longer.py --arch resnet18 --depth 18 --resume ckpt_oneshot_93/pruned.pth.tar --save_dir ckpt_ri_93 --gpu-id 2
python cifar_scratch_no_longer.py --arch resnet18 --depth 18 --resume ckpt_oneshot_94/pruned.pth.tar --save_dir ckpt_ri_94 --gpu-id 2
python cifar_scratch_no_longer.py --arch resnet18 --depth 18 --resume ckpt_oneshot_95/pruned.pth.tar --save_dir ckpt_ri_95 --gpu-id 2
python cifar_scratch_no_longer.py --arch resnet18 --depth 18 --resume ckpt_oneshot_96/pruned.pth.tar --save_dir ckpt_ri_96 --gpu-id 2
python cifar_scratch_no_longer.py --arch resnet18 --depth 18 --resume ckpt_oneshot_97/pruned.pth.tar --save_dir ckpt_ri_97 --gpu-id 2


# Teacher-Student ##############################
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_20/pruned.pth.tar --model ckpt/init.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kd_20 --gpu-id 0
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_36/pruned.pth.tar --model ckpt/init.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kd_36 --gpu-id 0
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_49/pruned.pth.tar --model ckpt/init.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kd_49 --gpu-id 0
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_59/pruned.pth.tar --model ckpt/init.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kd_59 --gpu-id 0
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_67/pruned.pth.tar --model ckpt/init.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kd_67 --gpu-id 0
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_74/pruned.pth.tar --model ckpt/init.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kd_74 --gpu-id 0
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_79/pruned.pth.tar --model ckpt/init.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kd_79 --gpu-id 0
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_83/pruned.pth.tar --model ckpt/init.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kd_83 --gpu-id 0
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_87/pruned.pth.tar --model ckpt/init.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kd_87 --gpu-id 0
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/init.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kd_89 --gpu-id 0

python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_91/pruned.pth.tar --model ckpt/init.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kd_91 --gpu-id 0
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_93/pruned.pth.tar --model ckpt/init.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kd_93 --gpu-id 0
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_94/pruned.pth.tar --model ckpt/init.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kd_94 --gpu-id 0
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_95/pruned.pth.tar --model ckpt/init.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kd_95 --gpu-id 0
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_96/pruned.pth.tar --model ckpt/init.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kd_96 --gpu-id 0
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_97/pruned.pth.tar --model ckpt/init.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kd_97 --gpu-id 0


# Rewind ##############################

# rewind point: 182 * 7% = 13
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/13_checkpoint.pth.tar --save_dir ckpt_rw_p89_e07 --epochs 169 --schedule 77 122 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_87/pruned.pth.tar --model ckpt/13_checkpoint.pth.tar --save_dir ckpt_rw_p87_e07 --epochs 169 --schedule 77 122 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_83/pruned.pth.tar --model ckpt/13_checkpoint.pth.tar --save_dir ckpt_rw_p83_e07 --epochs 169 --schedule 77 122 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_79/pruned.pth.tar --model ckpt/13_checkpoint.pth.tar --save_dir ckpt_rw_p79_e07 --epochs 169 --schedule 77 122 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_74/pruned.pth.tar --model ckpt/13_checkpoint.pth.tar --save_dir ckpt_rw_p74_e07 --epochs 169 --schedule 77 122 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_67/pruned.pth.tar --model ckpt/13_checkpoint.pth.tar --save_dir ckpt_rw_p67_e07 --epochs 169 --schedule 77 122 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_59/pruned.pth.tar --model ckpt/13_checkpoint.pth.tar --save_dir ckpt_rw_p59_e07 --epochs 169 --schedule 77 122 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_49/pruned.pth.tar --model ckpt/13_checkpoint.pth.tar --save_dir ckpt_rw_p49_e07 --epochs 169 --schedule 77 122 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_36/pruned.pth.tar --model ckpt/13_checkpoint.pth.tar --save_dir ckpt_rw_p36_e07 --epochs 169 --schedule 77 122 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_20/pruned.pth.tar --model ckpt/13_checkpoint.pth.tar --save_dir ckpt_rw_p20_e07 --epochs 169 --schedule 77 122 --gpu-id 2

python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_91/pruned.pth.tar --model ckpt/13_checkpoint.pth.tar --save_dir ckpt_rw_p91_e07 --epochs 169 --schedule 77 122 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_93/pruned.pth.tar --model ckpt/13_checkpoint.pth.tar --save_dir ckpt_rw_p93_e07 --epochs 169 --schedule 77 122 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_94/pruned.pth.tar --model ckpt/13_checkpoint.pth.tar --save_dir ckpt_rw_p94_e07 --epochs 169 --schedule 77 122 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_95/pruned.pth.tar --model ckpt/13_checkpoint.pth.tar --save_dir ckpt_rw_p95_e07 --epochs 169 --schedule 77 122 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_96/pruned.pth.tar --model ckpt/13_checkpoint.pth.tar --save_dir ckpt_rw_p96_e07 --epochs 169 --schedule 77 122 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_97/pruned.pth.tar --model ckpt/13_checkpoint.pth.tar --save_dir ckpt_rw_p97_e07 --epochs 169 --schedule 77 122 --gpu-id 2


# rewind point: 182 * 18% = 33
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --save_dir ckpt_rw_p89_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_87/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --save_dir ckpt_rw_p87_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_83/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --save_dir ckpt_rw_p83_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_79/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --save_dir ckpt_rw_p79_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_74/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --save_dir ckpt_rw_p74_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_67/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --save_dir ckpt_rw_p67_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_59/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --save_dir ckpt_rw_p59_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_49/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --save_dir ckpt_rw_p49_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_36/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --save_dir ckpt_rw_p36_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_20/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --save_dir ckpt_rw_p20_e18 --epochs 149 --schedule 57 102 --gpu-id 3

python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_91/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --save_dir ckpt_rw_p91_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_93/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --save_dir ckpt_rw_p93_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_94/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --save_dir ckpt_rw_p94_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_95/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --save_dir ckpt_rw_p95_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_96/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --save_dir ckpt_rw_p96_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_97/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --save_dir ckpt_rw_p97_e18 --epochs 149 --schedule 57 102 --gpu-id 3

# rewind point: 182 * 29% = 53
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/53_checkpoint.pth.tar --save_dir ckpt_rw_p89_e29 --epochs 129 --schedule 37 82 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_87/pruned.pth.tar --model ckpt/53_checkpoint.pth.tar --save_dir ckpt_rw_p87_e29 --epochs 129 --schedule 37 82 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_83/pruned.pth.tar --model ckpt/53_checkpoint.pth.tar --save_dir ckpt_rw_p83_e29 --epochs 129 --schedule 37 82 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_79/pruned.pth.tar --model ckpt/53_checkpoint.pth.tar --save_dir ckpt_rw_p79_e29 --epochs 129 --schedule 37 82 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_74/pruned.pth.tar --model ckpt/53_checkpoint.pth.tar --save_dir ckpt_rw_p74_e29 --epochs 129 --schedule 37 82 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_67/pruned.pth.tar --model ckpt/53_checkpoint.pth.tar --save_dir ckpt_rw_p67_e29 --epochs 129 --schedule 37 82 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_59/pruned.pth.tar --model ckpt/53_checkpoint.pth.tar --save_dir ckpt_rw_p59_e29 --epochs 129 --schedule 37 82 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_49/pruned.pth.tar --model ckpt/53_checkpoint.pth.tar --save_dir ckpt_rw_p49_e29 --epochs 129 --schedule 37 82 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_36/pruned.pth.tar --model ckpt/53_checkpoint.pth.tar --save_dir ckpt_rw_p36_e29 --epochs 129 --schedule 37 82 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_20/pruned.pth.tar --model ckpt/53_checkpoint.pth.tar --save_dir ckpt_rw_p20_e29 --epochs 129 --schedule 37 82 --gpu-id 3

python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_91/pruned.pth.tar --model ckpt/53_checkpoint.pth.tar --save_dir ckpt_rw_p91_e29 --epochs 129 --schedule 37 82 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_93/pruned.pth.tar --model ckpt/53_checkpoint.pth.tar --save_dir ckpt_rw_p93_e29 --epochs 129 --schedule 37 82 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_94/pruned.pth.tar --model ckpt/53_checkpoint.pth.tar --save_dir ckpt_rw_p94_e29 --epochs 129 --schedule 37 82 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_95/pruned.pth.tar --model ckpt/53_checkpoint.pth.tar --save_dir ckpt_rw_p95_e29 --epochs 129 --schedule 37 82 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_96/pruned.pth.tar --model ckpt/53_checkpoint.pth.tar --save_dir ckpt_rw_p96_e29 --epochs 129 --schedule 37 82 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_97/pruned.pth.tar --model ckpt/53_checkpoint.pth.tar --save_dir ckpt_rw_p97_e29 --epochs 129 --schedule 37 82 --gpu-id 3

# rewind point: 182 * 40% = 73
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/73_checkpoint.pth.tar --save_dir ckpt_rw_p89_e40 --epochs 109 --schedule 17 62 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_87/pruned.pth.tar --model ckpt/73_checkpoint.pth.tar --save_dir ckpt_rw_p87_e40 --epochs 109 --schedule 17 62 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_83/pruned.pth.tar --model ckpt/73_checkpoint.pth.tar --save_dir ckpt_rw_p83_e40 --epochs 109 --schedule 17 62 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_79/pruned.pth.tar --model ckpt/73_checkpoint.pth.tar --save_dir ckpt_rw_p79_e40 --epochs 109 --schedule 17 62 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_74/pruned.pth.tar --model ckpt/73_checkpoint.pth.tar --save_dir ckpt_rw_p74_e40 --epochs 109 --schedule 17 62 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_67/pruned.pth.tar --model ckpt/73_checkpoint.pth.tar --save_dir ckpt_rw_p67_e40 --epochs 109 --schedule 17 62 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_59/pruned.pth.tar --model ckpt/73_checkpoint.pth.tar --save_dir ckpt_rw_p59_e40 --epochs 109 --schedule 17 62 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_49/pruned.pth.tar --model ckpt/73_checkpoint.pth.tar --save_dir ckpt_rw_p49_e40 --epochs 109 --schedule 17 62 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_36/pruned.pth.tar --model ckpt/73_checkpoint.pth.tar --save_dir ckpt_rw_p36_e40 --epochs 109 --schedule 17 62 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_20/pruned.pth.tar --model ckpt/73_checkpoint.pth.tar --save_dir ckpt_rw_p20_e40 --epochs 109 --schedule 17 62 --gpu-id 2

python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_91/pruned.pth.tar --model ckpt/73_checkpoint.pth.tar --save_dir ckpt_rw_p91_e40 --epochs 109 --schedule 17 62 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_93/pruned.pth.tar --model ckpt/73_checkpoint.pth.tar --save_dir ckpt_rw_p93_e40 --epochs 109 --schedule 17 62 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_94/pruned.pth.tar --model ckpt/73_checkpoint.pth.tar --save_dir ckpt_rw_p94_e40 --epochs 109 --schedule 17 62 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_95/pruned.pth.tar --model ckpt/73_checkpoint.pth.tar --save_dir ckpt_rw_p95_e40 --epochs 109 --schedule 17 62 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_96/pruned.pth.tar --model ckpt/73_checkpoint.pth.tar --save_dir ckpt_rw_p96_e40 --epochs 109 --schedule 17 62 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_97/pruned.pth.tar --model ckpt/73_checkpoint.pth.tar --save_dir ckpt_rw_p97_e40 --epochs 109 --schedule 17 62 --gpu-id 2


# rewind point: 182 * 51% = 93
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/93_checkpoint.pth.tar --save_dir ckpt_rw_p89_e51 --epochs 89 --schedule  42 --lr 0.01 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_87/pruned.pth.tar --model ckpt/93_checkpoint.pth.tar --save_dir ckpt_rw_p87_e51 --epochs 89 --schedule  42 --lr 0.01 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_83/pruned.pth.tar --model ckpt/93_checkpoint.pth.tar --save_dir ckpt_rw_p83_e51 --epochs 89 --schedule  42 --lr 0.01 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_79/pruned.pth.tar --model ckpt/93_checkpoint.pth.tar --save_dir ckpt_rw_p79_e51 --epochs 89 --schedule  42 --lr 0.01 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_74/pruned.pth.tar --model ckpt/93_checkpoint.pth.tar --save_dir ckpt_rw_p74_e51 --epochs 89 --schedule  42 --lr 0.01 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_67/pruned.pth.tar --model ckpt/93_checkpoint.pth.tar --save_dir ckpt_rw_p67_e51 --epochs 89 --schedule  42 --lr 0.01 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_59/pruned.pth.tar --model ckpt/93_checkpoint.pth.tar --save_dir ckpt_rw_p59_e51 --epochs 89 --schedule  42 --lr 0.01 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_49/pruned.pth.tar --model ckpt/93_checkpoint.pth.tar --save_dir ckpt_rw_p49_e51 --epochs 89 --schedule  42 --lr 0.01 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_36/pruned.pth.tar --model ckpt/93_checkpoint.pth.tar --save_dir ckpt_rw_p36_e51 --epochs 89 --schedule  42 --lr 0.01 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_20/pruned.pth.tar --model ckpt/93_checkpoint.pth.tar --save_dir ckpt_rw_p20_e51 --epochs 89 --schedule  42 --lr 0.01 --gpu-id 2

python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_91/pruned.pth.tar --model ckpt/93_checkpoint.pth.tar --save_dir ckpt_rw_p91_e51 --epochs 89 --schedule  42 --lr 0.01 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_93/pruned.pth.tar --model ckpt/93_checkpoint.pth.tar --save_dir ckpt_rw_p93_e51 --epochs 89 --schedule  42 --lr 0.01 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_94/pruned.pth.tar --model ckpt/93_checkpoint.pth.tar --save_dir ckpt_rw_p94_e51 --epochs 89 --schedule  42 --lr 0.01 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_95/pruned.pth.tar --model ckpt/93_checkpoint.pth.tar --save_dir ckpt_rw_p95_e51 --epochs 89 --schedule  42 --lr 0.01 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_96/pruned.pth.tar --model ckpt/93_checkpoint.pth.tar --save_dir ckpt_rw_p96_e51 --epochs 89 --schedule  42 --lr 0.01 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_97/pruned.pth.tar --model ckpt/93_checkpoint.pth.tar --save_dir ckpt_rw_p97_e51 --epochs 89 --schedule  42 --lr 0.01 --gpu-id 2


# rewind point: 182 * 62% = 113
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/113_checkpoint.pth.tar --save_dir ckpt_rw_p89_e62 --epochs 69 --schedule  22 --lr 0.01 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_87/pruned.pth.tar --model ckpt/113_checkpoint.pth.tar --save_dir ckpt_rw_p87_e62 --epochs 69 --schedule  22 --lr 0.01 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_83/pruned.pth.tar --model ckpt/113_checkpoint.pth.tar --save_dir ckpt_rw_p83_e62 --epochs 69 --schedule  22 --lr 0.01 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_79/pruned.pth.tar --model ckpt/113_checkpoint.pth.tar --save_dir ckpt_rw_p79_e62 --epochs 69 --schedule  22 --lr 0.01 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_74/pruned.pth.tar --model ckpt/113_checkpoint.pth.tar --save_dir ckpt_rw_p74_e62 --epochs 69 --schedule  22 --lr 0.01 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_67/pruned.pth.tar --model ckpt/113_checkpoint.pth.tar --save_dir ckpt_rw_p67_e62 --epochs 69 --schedule  22 --lr 0.01 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_59/pruned.pth.tar --model ckpt/113_checkpoint.pth.tar --save_dir ckpt_rw_p59_e62 --epochs 69 --schedule  22 --lr 0.01 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_49/pruned.pth.tar --model ckpt/113_checkpoint.pth.tar --save_dir ckpt_rw_p49_e62 --epochs 69 --schedule  22 --lr 0.01 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_36/pruned.pth.tar --model ckpt/113_checkpoint.pth.tar --save_dir ckpt_rw_p36_e62 --epochs 69 --schedule  22 --lr 0.01 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_20/pruned.pth.tar --model ckpt/113_checkpoint.pth.tar --save_dir ckpt_rw_p20_e62 --epochs 69 --schedule  22 --lr 0.01 --gpu-id 3

python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_91/pruned.pth.tar --model ckpt/113_checkpoint.pth.tar --save_dir ckpt_rw_p91_e62 --epochs 69 --schedule  22 --lr 0.01 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_93/pruned.pth.tar --model ckpt/113_checkpoint.pth.tar --save_dir ckpt_rw_p93_e62 --epochs 69 --schedule  22 --lr 0.01 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_94/pruned.pth.tar --model ckpt/113_checkpoint.pth.tar --save_dir ckpt_rw_p94_e62 --epochs 69 --schedule  22 --lr 0.01 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_95/pruned.pth.tar --model ckpt/113_checkpoint.pth.tar --save_dir ckpt_rw_p95_e62 --epochs 69 --schedule  22 --lr 0.01 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_96/pruned.pth.tar --model ckpt/113_checkpoint.pth.tar --save_dir ckpt_rw_p96_e62 --epochs 69 --schedule  22 --lr 0.01 --gpu-id 3
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_97/pruned.pth.tar --model ckpt/113_checkpoint.pth.tar --save_dir ckpt_rw_p97_e62 --epochs 69 --schedule  22 --lr 0.01 --gpu-id 3


# rewind point: 182 * 73% = 133
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/133_checkpoint.pth.tar --save_dir ckpt_rw_p89_e73 --epochs 49 --schedule  2 --lr 0.01 --gpu-id 0
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_87/pruned.pth.tar --model ckpt/133_checkpoint.pth.tar --save_dir ckpt_rw_p87_e73 --epochs 49 --schedule  2 --lr 0.01 --gpu-id 0
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_83/pruned.pth.tar --model ckpt/133_checkpoint.pth.tar --save_dir ckpt_rw_p83_e73 --epochs 49 --schedule  2 --lr 0.01 --gpu-id 0
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_79/pruned.pth.tar --model ckpt/133_checkpoint.pth.tar --save_dir ckpt_rw_p79_e73 --epochs 49 --schedule  2 --lr 0.01 --gpu-id 0
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_74/pruned.pth.tar --model ckpt/133_checkpoint.pth.tar --save_dir ckpt_rw_p74_e73 --epochs 49 --schedule  2 --lr 0.01 --gpu-id 0
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_67/pruned.pth.tar --model ckpt/133_checkpoint.pth.tar --save_dir ckpt_rw_p67_e73 --epochs 49 --schedule  2 --lr 0.01 --gpu-id 0
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_59/pruned.pth.tar --model ckpt/133_checkpoint.pth.tar --save_dir ckpt_rw_p59_e73 --epochs 49 --schedule  2 --lr 0.01 --gpu-id 0
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_49/pruned.pth.tar --model ckpt/133_checkpoint.pth.tar --save_dir ckpt_rw_p49_e73 --epochs 49 --schedule  2 --lr 0.01 --gpu-id 0
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_36/pruned.pth.tar --model ckpt/133_checkpoint.pth.tar --save_dir ckpt_rw_p36_e73 --epochs 49 --schedule  2 --lr 0.01 --gpu-id 0
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_20/pruned.pth.tar --model ckpt/133_checkpoint.pth.tar --save_dir ckpt_rw_p20_e73 --epochs 49 --schedule  2 --lr 0.01 --gpu-id 0

python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_91/pruned.pth.tar --model ckpt/133_checkpoint.pth.tar --save_dir ckpt_rw_p91_e73 --epochs 49 --schedule  2 --lr 0.01 --gpu-id 0
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_93/pruned.pth.tar --model ckpt/133_checkpoint.pth.tar --save_dir ckpt_rw_p93_e73 --epochs 49 --schedule  2 --lr 0.01 --gpu-id 0
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_94/pruned.pth.tar --model ckpt/133_checkpoint.pth.tar --save_dir ckpt_rw_p94_e73 --epochs 49 --schedule  2 --lr 0.01 --gpu-id 0
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_95/pruned.pth.tar --model ckpt/133_checkpoint.pth.tar --save_dir ckpt_rw_p95_e73 --epochs 49 --schedule  2 --lr 0.01 --gpu-id 0
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_96/pruned.pth.tar --model ckpt/133_checkpoint.pth.tar --save_dir ckpt_rw_p96_e73 --epochs 49 --schedule  2 --lr 0.01 --gpu-id 0
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_97/pruned.pth.tar --model ckpt/133_checkpoint.pth.tar --save_dir ckpt_rw_p97_e73 --epochs 49 --schedule  2 --lr 0.01 --gpu-id 0


# rewind point: 182 * 84% = 153
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/153_checkpoint.pth.tar --save_dir ckpt_rw_p89_e84 --epochs 29 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_87/pruned.pth.tar --model ckpt/153_checkpoint.pth.tar --save_dir ckpt_rw_p87_e84 --epochs 29 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_83/pruned.pth.tar --model ckpt/153_checkpoint.pth.tar --save_dir ckpt_rw_p83_e84 --epochs 29 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_79/pruned.pth.tar --model ckpt/153_checkpoint.pth.tar --save_dir ckpt_rw_p79_e84 --epochs 29 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_74/pruned.pth.tar --model ckpt/153_checkpoint.pth.tar --save_dir ckpt_rw_p74_e84 --epochs 29 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_67/pruned.pth.tar --model ckpt/153_checkpoint.pth.tar --save_dir ckpt_rw_p67_e84 --epochs 29 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_59/pruned.pth.tar --model ckpt/153_checkpoint.pth.tar --save_dir ckpt_rw_p59_e84 --epochs 29 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_49/pruned.pth.tar --model ckpt/153_checkpoint.pth.tar --save_dir ckpt_rw_p49_e84 --epochs 29 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_36/pruned.pth.tar --model ckpt/153_checkpoint.pth.tar --save_dir ckpt_rw_p36_e84 --epochs 29 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_20/pruned.pth.tar --model ckpt/153_checkpoint.pth.tar --save_dir ckpt_rw_p20_e84 --epochs 29 --lr 0.001 --gpu-id 2

python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_91/pruned.pth.tar --model ckpt/153_checkpoint.pth.tar --save_dir ckpt_rw_p91_e84 --epochs 29 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_93/pruned.pth.tar --model ckpt/153_checkpoint.pth.tar --save_dir ckpt_rw_p93_e84 --epochs 29 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_94/pruned.pth.tar --model ckpt/153_checkpoint.pth.tar --save_dir ckpt_rw_p94_e84 --epochs 29 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_95/pruned.pth.tar --model ckpt/153_checkpoint.pth.tar --save_dir ckpt_rw_p95_e84 --epochs 29 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_96/pruned.pth.tar --model ckpt/153_checkpoint.pth.tar --save_dir ckpt_rw_p96_e84 --epochs 29 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_97/pruned.pth.tar --model ckpt/153_checkpoint.pth.tar --save_dir ckpt_rw_p97_e84 --epochs 29 --lr 0.001 --gpu-id 2


# rewind point: 182 * 95% = 173
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/173_checkpoint.pth.tar --save_dir ckpt_rw_p89_e95 --epochs 9 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_87/pruned.pth.tar --model ckpt/173_checkpoint.pth.tar --save_dir ckpt_rw_p87_e95 --epochs 9 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_83/pruned.pth.tar --model ckpt/173_checkpoint.pth.tar --save_dir ckpt_rw_p83_e95 --epochs 9 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_79/pruned.pth.tar --model ckpt/173_checkpoint.pth.tar --save_dir ckpt_rw_p79_e95 --epochs 9 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_74/pruned.pth.tar --model ckpt/173_checkpoint.pth.tar --save_dir ckpt_rw_p74_e95 --epochs 9 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_67/pruned.pth.tar --model ckpt/173_checkpoint.pth.tar --save_dir ckpt_rw_p67_e95 --epochs 9 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_59/pruned.pth.tar --model ckpt/173_checkpoint.pth.tar --save_dir ckpt_rw_p59_e95 --epochs 9 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_49/pruned.pth.tar --model ckpt/173_checkpoint.pth.tar --save_dir ckpt_rw_p49_e95 --epochs 9 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_36/pruned.pth.tar --model ckpt/173_checkpoint.pth.tar --save_dir ckpt_rw_p36_e95 --epochs 9 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_20/pruned.pth.tar --model ckpt/173_checkpoint.pth.tar --save_dir ckpt_rw_p20_e95 --epochs 9 --lr 0.001 --gpu-id 2

python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_91/pruned.pth.tar --model ckpt/173_checkpoint.pth.tar --save_dir ckpt_rw_p91_e95 --epochs 9 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_93/pruned.pth.tar --model ckpt/173_checkpoint.pth.tar --save_dir ckpt_rw_p93_e95 --epochs 9 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_94/pruned.pth.tar --model ckpt/173_checkpoint.pth.tar --save_dir ckpt_rw_p94_e95 --epochs 9 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_95/pruned.pth.tar --model ckpt/173_checkpoint.pth.tar --save_dir ckpt_rw_p95_e95 --epochs 9 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_96/pruned.pth.tar --model ckpt/173_checkpoint.pth.tar --save_dir ckpt_rw_p96_e95 --epochs 9 --lr 0.001 --gpu-id 2
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_97/pruned.pth.tar --model ckpt/173_checkpoint.pth.tar --save_dir ckpt_rw_p97_e95 --epochs 9 --lr 0.001 --gpu-id 2


# KD rewinding, rewind point: 182 * 18% = 33
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kdrw_p89_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_87/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kdrw_p87_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_83/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kdrw_p83_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_79/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kdrw_p79_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_74/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kdrw_p74_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_67/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kdrw_p67_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_59/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kdrw_p59_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_49/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kdrw_p49_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_36/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kdrw_p36_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_20/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kdrw_p20_e18 --epochs 149 --schedule 57 102 --gpu-id 3

python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_91/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kdrw_p91_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_93/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kdrw_p93_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_94/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kdrw_p94_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_95/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kdrw_p95_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_96/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kdrw_p96_e18 --epochs 149 --schedule 57 102 --gpu-id 3
python cifar_lt_kd.py --arch resnet18 --depth 18 --resume ckpt_oneshot_97/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kdrw_p97_e18 --epochs 149 --schedule 57 102 --gpu-id 3


