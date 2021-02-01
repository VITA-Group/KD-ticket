# KD rewinding on ImageNet
Here we use the KD rewinding technique to achieve the state-of-the-art pruning performance on ImageNet.
To train model on ImageNet, you need to download the dataset from their official [website](http://www.image-net.org/), 
and create two folders `train/` and `val`. 


The scripts are quite similar as these in CIFAR-10. 

- Firstly, we train the original unpruned network. 
~~~
python -u baseline.py --depth 50 --save_dir  ckpt --apex 1 --gpu 0 1 2 3
~~~
This script will store the initialization weights and weights after each training epoch at `ckpt/` for using later.

- Secondly, we use one shot pruning technique to prune the network to a specifict sparsity level.
For example, here we prune it to the sparsity at 89.3%, which means we have 10.7% weights remaining. 
When finishing this script, the pruned mask is stored at `ckpt_oneshot_89/pruned.pth.tar`. 
~~~
python prune.py --depth 50 --percent 0.89262 --resume ckpt/90_checkpoint.pth.tar --save_dir ckpt_oneshot_89 --apex 1 --gpu 0 1 2 3
~~~

- Thirdly, you can train our Kd rewinding based on the results of previous steps. 
~~~
python -u kd_ticket.py --depth 50 --resume ckpt_oneshot_36/pruned.pth.tar --model ckpt/6_checkpoint.pth.tar --teacher ckpt/90_checkpoint.pth.tar --save_dir ckpt_kdrw_36 --epochs 90 --schedule 30 60 80 --eskd 60 --apex 1 --gpu-id 0 1 2 3
~~~

- We also provide the script of the lottery ticket and weight rewinding for fair comparision.  
~~~
# lottery ticket
python lottery_ticket.py --depth 50 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/init.pth.tar --save_dir ckpt_lt_89 --apex 1 --gpu-id 0 1 2 3

# weight rewinding
python -u lottery_ticket.py --depth 50 --resume ckpt_oneshot_89/pruned.pth.tar --model ckpt/6_checkpoint.pth.tar --save_dir ckpt_rw_p89_e6 --epochs 84 --schedule 24 54 74 --apex 1 --gpu-id 0 1 2 3
~~~




