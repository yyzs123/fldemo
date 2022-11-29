source "D:/miniconda3/etc/profile.d/conda.sh"
hash -r
conda activate fldemo

python main.py --dataset fmnist --iid --uniform --modeltype cnn --rounds 50 --local_ep 5 --num_users 100 --frac 0.1 --gpu -1 --seed 5201205
