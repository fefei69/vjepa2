# Launch app/vjepa_hanoi training, one process per GPU.
#
#   srun python -m app.vjepa_hanoi.run --fname <yaml>     # inside sbatch: one task per GPU (see train.sbatch)
#   python -m app.vjepa_hanoi.run --fname <yaml>          # single process (1 GPU, or CPU for smoke tests)
#
# Differences from app/main.py: a crash propagates to the exit code (app/main.py never joins its children, so SLURM
# reports COMPLETED); each rank pins its own GPU whether or not SLURM binds GPUs per task (train.py's import-time
# CUDA_VISIBLE_DEVICES=$SLURM_LOCALID breaks under per-task binding); and the rendezvous port is derived from the job
# id instead of the fixed 37129 of src/utils/distributed.py, which collides when two jobs share a node.

import argparse
import logging
import os
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, required=True, help="config file")
    args = parser.parse_args()

    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    rank = int(os.environ.get("SLURM_PROCID", 0))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))

    # -- pin one GPU per process before CUDA is initialised
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
    if len(visible) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible[local_rank % len(visible)]
    os.environ.setdefault("MASTER_ADDR", "localhost")
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(20000 + int(os.environ.get("SLURM_JOB_ID", os.getpid())) % 20000)

    import torch
    import torch.distributed as dist

    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # the single visible device
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size
    )

    with open(args.fname, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    if rank == 0:
        Path(params["folder"]).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(params["folder"], "params-pretrain.yaml"), "w") as f:
            yaml.dump(params, f)
    dist.barrier()

    from app.vjepa_hanoi import train  # imported after pinning the GPU

    logging.getLogger().setLevel(logging.INFO if rank == 0 else logging.ERROR)
    train.main(args=params)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
