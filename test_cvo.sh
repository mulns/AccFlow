
##### CVO-clean pass #####
# Test direct methods
python test_cvo.py -d clean -acc direct -ofe raft --ofe_ckpt checkpoints/raft-things.pth
python test_cvo.py -d clean -acc direct -ofe raft --ofe_ckpt checkpoints/raft-cvo.pth
python test_cvo.py -d clean -acc direct -ofe gma --ofe_ckpt checkpoints/gma-things.pth
python test_cvo.py -d clean -acc direct -ofe gma --ofe_ckpt checkpoints/gma-cvo.pth

# Test accumulation methods
python test_cvo.py -d clean -acc acc -ofe raft --acc_ckpt checkpoints/acc+raft-things.pth
python test_cvo.py -d clean -acc acc -ofe raft --acc_ckpt checkpoints/acc+raft-cvo.pth
python test_cvo.py -d clean -acc acc -ofe gma --acc_ckpt checkpoints/acc+gma-things.pth
python test_cvo.py -d clean -acc acc -ofe gma --acc_ckpt checkpoints/acc+gma-cvo.pth

##### CVO-clean pass #####
# Test direct methods
python test_cvo.py -d final -acc direct -ofe raft --ofe_ckpt checkpoints/raft-things.pth
python test_cvo.py -d final -acc direct -ofe raft --ofe_ckpt checkpoints/raft-cvo.pth
python test_cvo.py -d final -acc direct -ofe gma --ofe_ckpt checkpoints/gma-things.pth
python test_cvo.py -d final -acc direct -ofe gma --ofe_ckpt checkpoints/gma-cvo.pth
# Test accumulation methods
python test_cvo.py -d final -acc acc -ofe raft --acc_ckpt checkpoints/acc+raft-things.pth
python test_cvo.py -d final -acc acc -ofe raft --acc_ckpt checkpoints/acc+raft-cvo.pth
python test_cvo.py -d final -acc acc -ofe gma --acc_ckpt checkpoints/acc+gma-things.pth
python test_cvo.py -d final -acc acc -ofe gma --acc_ckpt checkpoints/acc+gma-cvo.pth