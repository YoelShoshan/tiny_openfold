# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from Bio.SVDSuperimposer import SVDSuperimposer
import numpy as np
import torch
from typing import Union

def _superimpose_np(reference, coords):
    """
        Superimposes coordinates onto a reference by minimizing RMSD using SVD.

        Args:
            reference:
                [N, 3] reference array
            coords:
                [N, 3] array
        Returns:
            A tuple of [N, 3] superimposed coords and the final RMSD.
    """
    sup = SVDSuperimposer()
    sup.set(reference, coords)
    sup.run()
    rot_matrix, trans_matrix = sup.get_rotran()
    return sup.get_transformed(), sup.get_rms(), rot_matrix, trans_matrix


def _superimpose_single(reference, coords):
    reference_np = reference.detach().cpu().numpy()    
    coords_np = coords.detach().cpu().numpy()
    superimposed, rmsd, rot_matrix, trans_matrix = _superimpose_np(reference_np, coords_np)
    return coords.new_tensor(superimposed), coords.new_tensor(rmsd), torch.from_numpy(rot_matrix), torch.from_numpy(trans_matrix)


def _to_torch_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise Exception(f'_to_torch_tensor: only supporting convertion from np.ndarray but got type {type(x)}')

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    raise Exception(f'_to_numpy: only supporting convertion from torch tensor but got type {type(x)}')

def superimpose(
        reference:Union[np.ndarray,torch.tensor], 
        coords:Union[np.ndarray,torch.tensor], 
        mask:Union[np.ndarray,torch.tensor], 
        return_type:str = 'numpy',
        verbose:bool = False):
    """
        Superimposes coordinates onto a reference by minimizing RMSD using SVD.
        IMPORTANT: make sure you DO NOT provide shapes like [N, 14, 3] because it will then treat it as batches,
            which would result in getting separate superimposition of individual residues instead of the full structures

            For example, if what you have is [N,14,3] for both refererence and coords, you should use .reshape(-1, 3)
            and if you have [N,14] for mask, you should use .reshape(-1)

        Args:
            reference:
                [*, N, 3] reference tensor
            coords:
                [*, N, 3] tensor
            mask:
                [*, N] tensor        
        Returns:
            A tuple of [*, N, 3] superimposed coords and [*] final RMSDs.
    """
    if verbose:
        print(f'superimpose::debug:: reference {reference.shape} coords {coords.shape} mask {mask.shape}')
    if len(reference.shape)>2 and reference.shape[-2] in [14,15,37]:
        print(f'superimpose warning, you are using shape {reference.shape} which might mean you did not flatten the residues and superimposing will happen separately per residue which is likely not your intention')

    assert return_type in ['numpy', 'torch']

    reference = _to_torch_tensor(reference)
    coords = _to_torch_tensor(coords)
    mask = _to_torch_tensor(mask)
    
    def select_unmasked_coords(coords, mask):
        return torch.masked_select(
            coords,
            (mask > 0.)[..., None],
        ).reshape(-1, 3)

    batch_dims = reference.shape[:-2]
    flat_reference = reference.reshape((-1,) + reference.shape[-2:])
    flat_coords = coords.reshape((-1,) + reference.shape[-2:])
    flat_mask = mask.reshape((-1,) + mask.shape[-1:])
    superimposed_list = []
    rmsds = []
    rot_matrices = []
    trans_matrices = []
    for r, c, m in zip(flat_reference, flat_coords, flat_mask):
        r_unmasked_coords = select_unmasked_coords(r, m)
        c_unmasked_coords = select_unmasked_coords(c, m)
        superimposed, rmsd, rot_matrix, trans_matrix = _superimpose_single(
            r_unmasked_coords, 
            c_unmasked_coords
        )

        # This is very inelegant, but idk how else to invert the masking
        # procedure.
        count = 0
        superimposed_full_size = torch.zeros_like(r)
        for i, unmasked in enumerate(m):
            if(unmasked):
                superimposed_full_size[i] = superimposed[count]
                count += 1

        superimposed_list.append(superimposed_full_size)
        rmsds.append(rmsd)
        rot_matrices.append(rot_matrix)
        trans_matrices.append(trans_matrix)

    superimposed_stacked = torch.stack(superimposed_list, dim=0)
    rmsds_stacked = torch.stack(rmsds, dim=0)
    rot_matrices_stacked = torch.stack(rot_matrices, axis=0)
    trans_matrices_stacked = torch.stack(trans_matrices, axis=0)

    superimposed_reshaped = superimposed_stacked.reshape(
        batch_dims + coords.shape[-2:]
    )
    rmsds_reshaped = rmsds_stacked.reshape(
        batch_dims
    )

    if return_type == 'numpy':
        superimposed_reshaped = _to_numpy(superimposed_reshaped)
        rmsds_reshaped = _to_numpy(rmsds_reshaped)
        rot_matrices_stacked = _to_numpy(rot_matrices_stacked)
        trans_matrices_stacked = _to_numpy(trans_matrices_stacked)
    elif return_type == 'torch':
        superimposed_reshaped = _to_torch_tensor(superimposed_reshaped)
        rmsds_reshaped = _to_torch_tensor(rmsds_reshaped)
        rot_matrices_stacked = _to_torch_tensor(rot_matrices_stacked)
        trans_matrices_stacked = _to_torch_tensor(trans_matrices_stacked)

    if verbose:
        print('rmsds=', rmsds_reshaped)

    return superimposed_reshaped, rmsds_reshaped, rot_matrices_stacked, trans_matrices_stacked
