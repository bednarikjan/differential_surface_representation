""" Implementation of AtlasNet [1] with differential geometry properties based
    regularizers [2] for point-cloud auto-encoding task on ShapeNet.

[1] Groueix Thibault et al. AtlasNet: A Papier-Mâché Approach to Learning 3D
    Surface Generation. CVPR 2018.
[2] Bednarik Jan et al. Shape Reconstruction by Learning Differentiable Surface
    Representations. CoRR 2019.

Author: Jan Bednarik, jan.bednarik@epfl.ch
"""

# 3rd party
import torch
import torch.nn as nn

# Project files.
from helpers import Device
from encoder import ANEncoderPN
from decoder import DecoderMultiPatch, DecoderAtlasNet
from sampler import FNSamplerRandUniform
from diff_props import DiffGeomProps


class FoldingNetBase(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def _freeze_encoder(self, freeze=True):
        """ Freezes the trainable parameters of the encoder.

        Args:
            freeze (bool): Whether to freeze/unfreeze.
        """
        for ch in self.enc.children():
            for p in ch.parameters():
                p.requires_grad = not freeze


class FNDiffGeomPropsBase(FoldingNetBase, Device):
    """ Base class for models which compute 1st and 2nd order derivatives
    of predicted pointcloud wrt UV space.

    Args:
        alpha_chd (float): Weighting of CHD.
        gpu (bool): Whether to use GPU.
    """

    def __init__(self, fff=False, alpha_chd=1., gpu=True):
        FoldingNetBase.__init__(self)
        Device.__init__(self, gpu=gpu)

        self._reg_func_impl = self.register_dist_based

        # Diff. geom. props object.
        self.dgp = DiffGeomProps(
            normals=True, curv_mean=False, curv_gauss=False, fff=fff)

        # These quantities have to be computed in forward() pass.
        self.pc_pred = None
        self.geom_props = None

        # Loss weighting coeffs.
        self._alpha_chd = alpha_chd

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def distance_matrix(pc_N, pc_M):
        """ Computes a distance matrix between two pclouds.

        Args:
            pc_N (torch.Tensor): GT pcloud, shape (B, N, 3)
            pc_M (torch.Tensor): Predicted pcloud, shape (B, M, 3)

        Returns:
            Distance matrix, shape (B, M, N).
        """
        # Get per-point distance matrix.
        B, M, D = pc_M.shape
        B2, N, D2 = pc_N.shape

        assert (B == B2)
        assert (D == D2)
        assert (D == 3)

        x = pc_M.reshape((B, M, 1, D))
        y = pc_N.reshape((B, 1, N, D))

        return (x - y).pow(2).sum(dim=3).sqrt()  # (B, M, N, 3) -> (B, M, N)

    @staticmethod
    def register_dist_based(pc_gt, pc_p, *args):
        """

        Args:
            pc_gt:
            pc_p:

        Returns:

        """
        distm = FNDiffGeomPropsBase.distance_matrix(pc_gt, pc_p)  # (B, M, N)
        inds_p2gt = distm.argmin(dim=2)  # (B, M)
        inds_gt2p = distm.argmin(dim=1)  # (B, N)
        return inds_p2gt, inds_gt2p

    @staticmethod
    def echd(pc_gt, pc_p, inds_p2gt, inds_gt2p):
        """ Extended Chamfer distance.

        Args:
            pc_gt: (B, N, 3)
            pc_p: (B, M, 3)
            inds_p2gt: (B, M)
            inds_gt2p: (B, N)

        Returns:

        """
        # Reshape inds.
        inds_p2gt = inds_p2gt.unsqueeze(2).expand(-1, -1, 3)
        inds_gt2p = inds_gt2p.unsqueeze(2).expand(-1, -1, 3)

        # Get registered points.
        pc_gt_reg = pc_gt.gather(1, inds_p2gt)  # (B, M, 3)
        pc_p_reg = pc_p.gather(1, inds_gt2p)  # (B, N, 3)

        # Compute per-point-pair squared L2 distances.
        d_p2gt = (pc_p - pc_gt_reg).pow(2).sum(dim=2)  # (B, M)
        d_gt2p = (pc_gt - pc_p_reg).pow(2).sum(dim=2)  # (B, N)

        # Compute scalar loss.
        chd = d_p2gt.mean() + d_gt2p.mean()
        return chd

    def loss(self, pc_gt):
        """ Loss functions computing extended chamfer distance (eCHD).

        Args:
            pc_gt (torch.Tensor): GT pcloud, shape (B, N, 3).

        Returns:
            dict of torch.Tensor: Losses:
                - loss_tot: Total loss.
                - loss_chd: CHD loss.
        """
        losses = {}

        # Get registration.
        inds_p2gt, inds_gt2p = self._reg_func_impl(pc_gt, self.pc_pred)

        # Get echd loss.
        loss_chd = self.echd(pc_gt, self.pc_pred, inds_p2gt, inds_gt2p)
        losses['loss_chd'] = loss_chd

        # Total loss.
        loss_tot = loss_chd * self._alpha_chd
        losses['loss_tot'] = loss_tot
        return losses


class MultipatchDecoder(FNDiffGeomPropsBase):
    """ Base class for models with multipatch decoder conceptually inspired
    by Atlasnet [1].

    [1] T. Groueix et al. AtlasNet: A Papier-Mache Approach to Learning 3D
        Surface Generation. CVPR 2018.

    Args:
        M (int): # sampled points from UV space.
        num_patches (int): # patches, i.e. UV parameterizations.
        loss_scaled_isometry (bool): Whether to employ scaled isometry loss.
        alphas_sciso (dict): Hyperparams. of scaled isometry loss.
    """

    def __init__(self, M, num_patches, alpha_chd=1., loss_scaled_isometry=False,
                 alpha_scaled_isometry=0., alphas_sciso=None, gpu=True):
        super(MultipatchDecoder, self).__init__(
            fff=loss_scaled_isometry, alpha_chd=alpha_chd, gpu=gpu)

        self._num_patches = num_patches
        self._spp = M // num_patches  # Number of samples per patch.
        self._M = self._spp * num_patches
        self._loss_scaled_isometry = loss_scaled_isometry
        self._alpha_scaled_isometry = alpha_scaled_isometry

        self._loss_sciso = torch.tensor(0.).to(self.device)
        self._zero = torch.tensor(0.).to(self.device)
        self._one = torch.tensor(1.).to(self.device)
        self._mone = torch.tensor(-1.).to(self.device)
        self._eps = torch.tensor(1e-20)

        if loss_scaled_isometry:
            self._alphas_si = {k: torch.tensor(float(v)).to(self.device)
                               for k, v in alphas_sciso.items()}

        # Auxiliary variables.
        self._loss_iters = 0

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def collapsed_patches_A(self, max_ratio=1e-3, collapsed=True):
        """ Detects the collapsed patches by inspecting the ratios of their
        areas which are computed analytically.

        Returns:
            list of torch.Tensor of int32: Per within-batch sample indices
                corresponding to the collapsed patches (or non-collapsed if
                `collapsed=False`), shape (P, ) for evey item in list of
                length B, P is # patches.
        """
        E, F, G = self.geom_props['fff']. \
            reshape((-1, self._num_patches, self._spp, 3)). \
            permute(3, 0, 1, 2)  # Each (B, P, spp)
        Ap = (E * G - F ** 2).mean(dim=2).detach()  # (B, P)
        mu_Ap = Ap.mean(dim=1, keepdim=True)  # (B, 1)
        inds = (Ap / (mu_Ap + 1e-30)) < max_ratio  # (B, P), uint8
        inds = inds if collapsed else ~inds
        return [s.nonzero().reshape((-1,)).type(torch.int32) for s in inds]

    def loss_clps_olap(self, **kwargs):
        """

        Args:
            **kwargs: Must contain 'areas', per sample GT area, torch.Tensor
                of shape (B, ).
        """
        B = self.pc_pred.shape[0]

        # Get first fundamental form.
        E, F, G = self.geom_props['fff']. \
            reshape((B, self._num_patches, self._spp, 3)). \
            permute(3, 0, 1, 2)  # Each (B, P, spp)

        # Get per point local squared area.
        A2 = torch.max(E * G - F.pow(2), self._zero)  # (B, P, spp)
        A = A2.sqrt()  # (B, P, spp)

        muE = E.mean()
        muG = G.mean()

        L_stretch = ((E - G).pow(2) / (A2 + self._eps)).mean() * \
                    self._alphas_si['stretch']
        L_E = ((E - muE).pow(2) / (A2 + self._eps)).mean() * \
              self._alphas_si['E']
        L_G = ((G - muG).pow(2) / (A2 + self._eps)).mean() * \
              self._alphas_si['G']
        L_F = (F.pow(2) / (A2 + self._eps)).mean() * self._alphas_si['skew']

        # Loss total area.
        A_gt = kwargs['areas'] * self._alphas_si['total_area_mult']
        L_Atot = torch.max(self._zero, A.mean(dim=2).sum(dim=1) - A_gt). \
                     pow(2).mean() * self._alphas_si['total_area']

        return {'L_skew': L_F, 'L_E': L_E, 'L_G': L_G,
                'L_stretch': L_stretch, 'L_Atot': L_Atot,
                'loss_sciso': L_F + L_E + L_G + L_stretch + L_Atot}

    def loss(self, pc_gt, normals_gt=None, curvm_gt=None, curvg_gt=None,
             mask_nonbound_verts=None, areas_gt=None):
        """ Computes the loss as a combination of CHD, normals loss, curvature
        loss and patch collapse loss.

        pc_gt (torch.Tensor): GT pcloud, shape (B, N, 3).
        areas_gt (torch.Tensor): Per sample surface area [m^2].

        Returns:
            dict of torch.Tensor:
                - loss_tot: Total loss.
                - loss_chd: CHD loss.
                - loss_sciso: Scaled isometry loss.
        """
        # Get CHF, normals, curv., fmo loss.
        losses = FNDiffGeomPropsBase.loss(self, pc_gt)

        # Get scaled isometry loss.
        loss_sciso = self._loss_sciso
        if self._loss_scaled_isometry:
            losses_sciso = self.loss_clps_olap(areas=areas_gt)
            losses.update(**losses_sciso)
            loss_sciso = losses_sciso['loss_sciso']

        # Total loss.
        losses['loss_tot'] += self._alpha_scaled_isometry * loss_sciso
        return losses


class AtlasNetReimpl(MultipatchDecoder):
    """ Re-implementatio of the original atlasnet for auttencoding (AE) task.
    It is possible to change the activation function of the decoder layers.
    `dec_activ_fns` changes all but last activations, `dec_use_tanh` chooses
    whether to use tanh or linear as a last activation. It is possible to
    add scaled_isometry loss using `loss_scaled_isometry` and parameterize it
    using `alpha_scaled_isometry` and `alphas_sciso`.
    """
    def __init__(self, M=2500, code=1024, num_patches=1,
                 normalize_cw=False, freeze_encoder=False,
                 enc_load_weights=None, dec_activ_fns='relu',
                 dec_use_tanh=True, dec_batch_norm=True,
                 loss_scaled_isometry=False,
                 alpha_scaled_isometry=0., alphas_sciso=None, gpu=True,
                 **kwargs):
        MultipatchDecoder.__init__(
            self, M, num_patches, loss_scaled_isometry=loss_scaled_isometry,
            alpha_scaled_isometry=alpha_scaled_isometry,
            alphas_sciso=alphas_sciso, gpu=gpu)
        Device.__init__(self, gpu)

        self._code = code

        self.enc = ANEncoderPN(code, normalize_cw=normalize_cw, gpu=gpu)
        self.sampler = FNSamplerRandUniform((0., 1.), (0., 1.), M, gpu=gpu)
        self.dec = DecoderMultiPatch(
            num_patches, DecoderAtlasNet, code=code, sample_dim=2,
            batch_norm=dec_batch_norm, activ_fns=dec_activ_fns,
            use_tanh=dec_use_tanh, gpu=gpu, **kwargs)

        # Load encoder weights.
        if enc_load_weights is not None:
            self.load_state_dict(torch.load(enc_load_weights), strict=False)
            print('[INFO] Loaded weights for PointNet encoder from {}'.
                  format(enc_load_weights))

        # Freeze encoder.
        if freeze_encoder:
            self._freeze_encoder(freeze=True)

    def forward(self, x, **kwargs):
        B = x.shape[0]  # Batch size.
        spp = self._spp

        self.uv = self.sampler(B)
        self.uv.requires_grad = True

        # Get CWs from encoder.
        cws = self.enc(x)  # (B, code)

        # Get per-patch pcloud prediction.
        outs = []  # Each (B, spp, 3)
        for i in range(0, self._num_patches):
            grid = self.uv[:, i * spp:(i + 1) * spp]  # (B, spp, 2)
            y = cws.unsqueeze(1).expand(B, spp, self._code).contiguous()
            y = torch.cat([grid, y], 2).contiguous()  # (B, spp, code + 2)
            outs.append(self.dec[i](y, **kwargs))
        self.pc_pred = torch.cat(outs, 1).contiguous()  # (B, M, 3)

        # Get diff. geom. props.
        self.geom_props = self.dgp(self.pc_pred, self.uv)
