import argparse


def build_flow_estimator(name):
    if "raft" in name.lower():
        from .raft.raft import RAFT

        raft_args = argparse.Namespace(small=False, mixed_precision=True)
        model = RAFT(raft_args)
        return model
    elif "gma" in name.lower():
        from .gma.gma import RAFTGMA

        gma_args = argparse.Namespace(
            num_heads=1,
            mixed_precision=True,
            position_only=False,
            position_and_content=False,
        )
        model = RAFTGMA(gma_args)
        return model
    else:
        raise NotImplementedError("not supported yet..")
