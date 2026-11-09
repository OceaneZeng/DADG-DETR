"""Real two-process regression for variable-length training evidence."""
import os
from pathlib import Path
import socket
import subprocess
import sys


WORKER = r'''
import torch
import torch.distributed as dist
from util.training_runtime import gather_evidence
from models.graph_local.discovery import proposal_focal_loss, matched_known_ranking_loss
dist.init_process_group('gloo')
rank = dist.get_rank()
for sizes in ((0, 3), (2, 5), (0, 0)):
    count = sizes[rank]
    evidence = (torch.arange(count * 4, dtype=torch.float32).reshape(count, 4) + 100 * rank,
                torch.full((count,), rank - 1, dtype=torch.long),
                torch.arange(count, dtype=torch.float32) / 10)
    reference = [None, None]
    dist.all_gather_object(reference, evidence)
    actual = gather_evidence(evidence)
    for field in range(3):
        torch.testing.assert_close(actual[field], torch.cat([part[field] for part in reference]),
                                   rtol=0, atol=0)
# One rank has no selected objects. The DDP mean of local gradients must
# equal the global-positive-normalized objective over the concatenated batch.
joined = torch.tensor([[.7, -.4, 1.1], [.4, -.8, -2.]], requires_grad=True)
foreground = torch.tensor([[True, True, False], [False, False, False]])
weights = torch.tensor([[1., .2, 1.], [1., 1., 1.]])
reference_loss = proposal_focal_loss(joined, foreground, weights)
reference_grad = torch.autograd.grad(reference_loss, joined)[0]
local = joined.detach()[rank:rank + 1].clone().requires_grad_(True)
loss = proposal_focal_loss(local, foreground[rank:rank + 1], weights[rank:rank + 1])
grad = torch.autograd.grad(loss, local)[0]
torch.testing.assert_close(grad / 2, reference_grad[rank:rank + 1])
# Ranking must also handle a single GT in the entire world and an empty world.
for counts in ((3, 0), (1, 0), (0, 0), (1, 2)):
    joined = torch.linspace(-3, 3, sum(counts), requires_grad=True)
    reference_loss = torch.nn.functional.softplus(joined).sum() / max(sum(counts), 1)
    reference_grad = torch.autograd.grad(reference_loss, joined)[0]
    start = sum(counts[:rank])
    local = joined.detach()[start:start + counts[rank]].clone().requires_grad_(True)
    loss = matched_known_ranking_loss(local)
    grad = torch.autograd.grad(loss, local)[0]
    torch.testing.assert_close(grad / 2, reference_grad[start:start + counts[rank]])
    reduced_loss = loss.detach().clone()
    dist.all_reduce(reduced_loss)
    torch.testing.assert_close(reduced_loss / 2, reference_loss.detach())
dist.destroy_process_group()
'''


def test_two_rank_gather_preserves_ragged_empty_and_rank_order():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    processes = []
    try:
        for rank in range(2):
            env = dict(os.environ, MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port),
                       WORLD_SIZE='2', RANK=str(rank), OMP_NUM_THREADS='1',
                       CUDA_VISIBLE_DEVICES='', USE_LIBUV='0')
            processes.append(subprocess.Popen(
                [sys.executable, '-c', WORKER], cwd=Path(__file__).resolve().parents[3],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace'))
        for process in processes:
            output, _ = process.communicate(timeout=45)
            assert process.returncode == 0, output
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
