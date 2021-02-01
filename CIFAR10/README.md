# KD ticket and KD rewinding on CIFAR-10. 
We train ResNet-56, ResNet-110 and the standard ResNet-18 on CIFAR-10.
Since our paper includes lots of experiments, 
we save command for all experiments at 
[exp_res56_oneshot.sh](exp_res56_oneshot.sh), 
[exp_res110_oneshot.sh](exp_res110_oneshot.sh), and 
[exp_res18_oneshot.sh](exp_res18_oneshot.sh). 

Here we talk the experiments on ResNet-56 as an example to illustrate it. 
You can see the whole scripts for it in [exp_res56_oneshot.sh](exp_res56_oneshot.sh). 

- Firstly, we train the original unpruned network. 
~~~
python cifar_baseline.py --arch oresnet --depth 56 --save_dir  ckpt/ --gpu-id 3
~~~
This script will store the initialization weights and weights after each training epoch at `ckpt/` for using later.

- Secondly, we use one shot pruning technique to prune the network to a specifict sparsity level.
For example, here we prune it to the sparsity at 89.3%, which means we have 10.7% weights remaining. 
When finishing this script, the pruned mask is stored at `ckpt_oneshot_89/pruned.pth.tar`. 
~~~
python cifar_prune.py --arch oresnet --depth 56 --percent 0.89262 --resume ckpt/182_checkpoint.pth.tar --save_dir ckpt_oneshot_89
~~~

- Thirdly, you can train our KD ticket or Kd rewinding based on the results of previous steps. 

~~~
# KD ticket
python cifar_lt_kd.py --arch oresnet --depth 56 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/init.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kd_89 --gpu-id 0

# KD rewinding
python cifar_lt_kd.py --arch oresnet --depth 56 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --teacher ckpt/182_checkpoint.pth.tar --save_dir ckpt_kdrw_p89_e18 --epochs 149 --schedule 57 102 --gpu-id 0
~~~

- We also provide the script of the lottery ticket and weight rewinding for fair comparision. Here we rewind to the weight at 18% training progress, which is the 33-rd epoch of the 182 epochs. 
Note that since we only re-train 182-33=149 epochs, we also need to change the hyperparameters for our LR scheduler. 
~~~
# lottery ticket
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_89 --gpu-id 3

# weight rewinding
python lottery_ticket.py --arch resnet18 --depth 18 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/33_checkpoint.pth.tar --save_dir ckpt_rw_p89_e18 --epochs 149 --schedule 57 102 --gpu-id 0
~~~




