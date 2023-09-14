python test_sintel.py -acc direct -ofe raft-things --ofe_ckpt checkpoints/raft-things.pth
python test_sintel.py -acc direct -ofe raft-cvo --ofe_ckpt checkpoints/raft-cvo.pth
python test_sintel.py -acc direct -ofe gma-things --ofe_ckpt checkpoints/gma-things.pth
python test_sintel.py -acc direct -ofe gma-cvo --ofe_ckpt checkpoints/gma-cvo.pth
python test_sintel.py -acc acc -ofe raft-things --acc_ckpt checkpoints/acc+raft-things.pth
python test_sintel.py -acc acc -ofe raft-cvo --acc_ckpt checkpoints/acc+raft-cvo.pth
python test_sintel.py -acc acc -ofe gma-things --acc_ckpt checkpoints/acc+gma-things.pth
python test_sintel.py -acc acc -ofe gma-cvo --acc_ckpt checkpoints/acc+gma-cvo.pth