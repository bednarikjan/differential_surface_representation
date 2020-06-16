""" This script computes the area for those meshes in ShapeNet which are
used by AtlasNeta and which are used to train [1].

[1] Bednarik Jan et al. Shape Reconstruction by Learning Differentiable Surface
    Representations. CVPR 2020.

Author: Jan Bednarik, jan.bednarik@epfl.ch
"""

# Project files.
from helpers import ls, lsd, jn, load_obj, mesh_area

# Python std.
import os
import sys
import time
import multiprocessing as mp
from multiprocessing import Value
from timeit import default_timer as timer

# 3rd party.
import numpy as np

# Settings.
pth_root_an = '.../AtlasNet/data/customShapeNet'  # Path to AtlasNet's ShapeNet
pth_root_sn = '.../shapenet/ShapeNetCore.v2'  # Path to original ShapeNet

num_processes = 20

# Shared var, # processed samples.
num_samples_done = Value('i', 0)
finished_seqs = Value('i', 0)


# Helpers.
def get_total_num_samples():
    num_total = 0
    seqs = lsd(pth_root_an)
    for s in seqs:
        num_total += len(ls(jn(pth_root_an, s), exts='txt'))
    return num_total


def load_tf(pth):
    with open(pth, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 2
        T = np.array([float(v) for v in lines[0].split()],
                     dtype=np.float32)
        s = float(lines[1])
    return T, s


def process_sequence(obj):
    # Get list of paths in obj. category in AtlasNet.
    pth_files_an = jn(pth_root_an, obj, 'ply')
    files = ls(pth_files_an, exts='txt')

    # Peocess files.
    for f in files:
        # Extract file name.
        fn_base = f.split('.')[0]

        # Load .obj mesh from ShapeNet.
        pth_f_sn = jn(
            pth_root_sn, obj, fn_base, 'models', 'model_normalized.obj')
        assert os.path.exists(pth_f_sn)
        verts, faces = load_obj(pth_f_sn)

        # Load tf and apply.
        pth_f_an = jn(pth_files_an, f)
        T, s = load_tf(pth_f_an)
        verts = (verts - T) / s

        # Compute area.
        area = mesh_area(verts, faces)

        # Write area to the file.
        with open(pth_f_an, 'r') as fobj:
            txt = fobj.read()
            assert len(txt.splitlines()) == 2
            has_nl = txt.endswith('\n')

        with open(pth_f_an, 'a') as fobj:
            fobj.write('{}{:.6f}'.format(('\n', '')[has_nl], area))

        with num_samples_done.get_lock():
            num_samples_done.value += 1

    with finished_seqs.get_lock():
        finished_seqs.value += 1


# Get object categories which AtlasNet uses.
objs = lsd(pth_root_an)
num_objs = len(objs)
num_samples_total = get_total_num_samples()

# Start processes.
proc_pool = mp.Pool(processes=num_processes)
for seq in objs:
    proc_pool.apply_async(process_sequence, args=(seq, ))

# Wait for all sequences to be finished.
tstart = timer()
while True:
    with num_samples_done.get_lock():
        nsd = num_samples_done.value

    t_elapsed = timer() - tstart
    eta = (num_samples_total - nsd) * (t_elapsed / (1e-6, nsd)[nsd > 0])
    print('\rProcessed samples {}/{}, t: {:.2f} m, ETA: {:.2f} m'.
          format(nsd, num_samples_total, t_elapsed / 60., eta / 60.),
          end='', file=sys.stderr)

    with finished_seqs.get_lock():
        if finished_seqs.value >= num_objs:
            assert(finished_seqs.value == num_objs)
            break
    time.sleep(2.0)
