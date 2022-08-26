import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.ssim import ssim_loss
from .modules.ms_ssim import MS_SSIM_Loss
from .modules.reconstruction import ReconstrLoss
from .modules.depth_smooth import inverse_depth_smoothness_loss

from .rendering_depth import get_render_srcdepth
from .rendering_img import get_rendered_src, get_rendered_ref


class UnSupLoss(nn.Module):
    def __init__(self):
        super(UnSupLoss, self).__init__()

        self.ssim_loss = ssim_loss
        self.reconstr_loss = ReconstrLoss(alpha=0.5)
        self.ms_ssim_loss = MS_SSIM_Loss(compensation=1)
        self.smooth_loss = inverse_depth_smoothness_loss

        self.debug_outputs = {}
    
    def calc_ref_loss_stage(self, depth, gt_imgs, gt_masks,
                            syn_src_imgs, syn_src_masks, smooth_ref_img):
        dtype = depth.dtype
        num_src_views = syn_src_imgs.size(1)
        k = min(num_src_views, 3)

        reconstr_loss_volume = []
        for idx in range(num_src_views):
            gt_img, gt_mask = gt_imgs[:, idx+1], gt_masks[:, idx+1]
            syn_img, syn_mask = syn_src_imgs[:, idx], syn_src_masks[:, idx]
            mask = (gt_mask.to(dtype) * syn_mask.to(dtype)).unsqueeze(1)

            reconstr_loss_ = self.reconstr_loss(syn_img * mask, gt_img * mask)
            reconstr_loss_volume.append(reconstr_loss_ + 1e4 * (1 - mask))
        reconstr_loss_volume = torch.stack(reconstr_loss_volume, 1)
        top_vals, _ = torch.topk(torch.neg(reconstr_loss_volume), k=k, sorted=False)
        top_vals = torch.neg(top_vals)
        top_mask = (top_vals < 1e4).to(torch.float32)
        top_vals = torch.mul(top_vals, top_mask)
        reconstr_loss_val = torch.mean(torch.sum(top_vals, dim=-1))

        total_ssim_loss_val = torch.tensor(
            0.0, dtype=torch.float32, device=gt_imgs.device, requires_grad=False
        )
        for idx in range(k-1):
            gt_img, gt_mask = gt_imgs[:, idx+1], gt_masks[:, idx+1].unsqueeze(1)
            syn_img, syn_mask = syn_src_imgs[:, idx], syn_src_masks[:, idx].unsqueeze(1)
            mask = gt_mask.to(torch.float32) * syn_mask.to(torch.float32)
            total_ssim_loss_val += self.ssim_loss(gt_img * mask, syn_img * mask, reduction='mean')
        total_ssim_loss_val = total_ssim_loss_val / (k - 1)

        # smooth loss
        gt_mask = gt_masks[:, 0].unsqueeze(1)
        smooth_loss_val = self.smooth_loss(depth * gt_mask, smooth_ref_img * mask, reduction='mean')

        return total_ssim_loss_val, reconstr_loss_val , smooth_loss_val

    def calc_src_loss_stage(self, src_depths, gt_imgs, gt_masks,
                            syn_ref_imgs, syn_ref_masks):
        dtype = src_depths.dtype
        num_src_views = src_depths.size(1)
        k = min(num_src_views, 3)

        reconstr_loss_volume = []
        gt_img, gt_mask = gt_imgs[:, 0], gt_masks[:, 0]
        for idx in range(num_src_views):
            syn_img, syn_mask = syn_ref_imgs[:, idx], syn_ref_masks[:, idx]
            mask = (gt_mask.to(dtype) * syn_mask.to(dtype)).unsqueeze(1)
            reconstr_loss_ = self.reconstr_loss(syn_img * mask, gt_img * mask)
            reconstr_loss_volume.append(reconstr_loss_ + 1e4 * (1 - mask))
        reconstr_loss_volume = torch.stack(reconstr_loss_volume, 1)
        top_vals, _ = torch.topk(torch.neg(reconstr_loss_volume), k=k, sorted=False)
        top_vals = torch.neg(top_vals)
        top_mask = (top_vals < 1e4).to(torch.float32)
        top_vals = torch.mul(top_vals, top_mask)
        reconstr_loss_val = torch.mean(torch.sum(top_vals, dim=-1))

        total_ssim_loss_val = torch.tensor(
            0.0, dtype=torch.float32, device=gt_imgs.device, requires_grad=False
        )
        gt_img, gt_mask = gt_imgs[:, 0], gt_masks[:, 0].unsqueeze(1)
        for idx in range(k-1):
            syn_img, syn_mask = syn_ref_imgs[:, idx], syn_ref_masks[:, idx].unsqueeze(1)
            mask = gt_mask.to(torch.float32) * syn_mask.to(torch.float32)
            total_ssim_loss_val += self.ssim_loss(gt_img * mask, syn_img * mask, reduction='mean')
        total_ssim_loss_val = total_ssim_loss_val / (k - 1)
        
        # smooth loss
        total_smooth_loss_val = torch.tensor(
            0.0, dtype=torch.float32, device=gt_imgs.device, requires_grad=False
        )
        # for idx in range(k-1):
        #     depth = src_depths[:, idx].unsqueeze(1)
        #     gt_img, gt_mask = gt_imgs[:, idx+1], gt_masks[:, idx+1].unsqueeze(1)
        #     total_smooth_loss_val += self.smooth_loss(depth * gt_mask, gt_img * mask, reduction='mean')
        # total_smooth_loss_val = total_smooth_loss_val / (k - 1)
        
        return total_ssim_loss_val, reconstr_loss_val, total_smooth_loss_val

    def calc_loss_stage(
        self, gt_imgs, gt_masks, cameras, stage_key,
        ref_depth, depth_values, prob_volume, min_depth, max_depth,
        ssim_loss_weight=12, reconstr_loss_weight=6,
        smooth_loss_weight=0.05, depth_consistency_weight=0.01):

        dtype = gt_imgs.dtype
        num_views = gt_imgs.size(1)
        batch, _, height, width = ref_depth.shape

        raw_height, raw_width = gt_imgs.shape[-2:]
        gt_imgs = gt_imgs.view(batch * num_views, 3, raw_height, raw_width)
        gt_imgs = F.interpolate(gt_imgs, (height, width), mode='nearest')
        gt_imgs = gt_imgs.view(batch, num_views, 3, height, width)
        gt_masks = gt_masks.view(batch * num_views, 1, raw_height, raw_width)
        gt_masks = F.interpolate(gt_masks, (height, width), mode='nearest')
        gt_masks = gt_masks.view(batch, num_views, height, width)
        
        # synthesize source depth, render refererce and souce images
        syn_src_depths, syn_src_depth_masks = get_render_srcdepth(
            depth_values, prob_volume, cameras, min_depth, max_depth
        )
        rendered_src_imgs, rendered_src_depths, rendered_src_masks = get_rendered_src(
            gt_imgs[:, 0], ref_depth, cameras, max_depth.max()
        )
        rendered_ref_imgs, rendered_ref_masks = get_rendered_ref(
            gt_imgs[:, 1:], syn_src_depths, syn_src_depth_masks, cameras, max_depth.max()
        )

        # smooth reference image, with synthesized reference images
        cur_masks = rendered_ref_masks.detach().to(dtype)
        smooth_ref_img = rendered_ref_imgs.detach() * cur_masks.unsqueeze(2)
        smooth_ref_img = torch.div(
            torch.sum(smooth_ref_img, dim=1),
            torch.sum(cur_masks, dim=1, keepdim=True) + 1e-6
        )
        smooth_ref_img = 0.5 * torch.where(
            torch.sum(cur_masks, dim=1, keepdim=True) > 0,
            smooth_ref_img, gt_imgs[:, 0]
        ) + 0.5 * gt_imgs[:, 0]

        # reference stage loss, predicted depth -> source images
        # ssim loss, reconstr loss for synthesized souce images
        # smooth loss for reference predicted images
        ref_ssim_loss, ref_reconstr_loss, ref_smooth_loss = self.calc_ref_loss_stage(
            ref_depth, gt_imgs, gt_masks, rendered_src_imgs, rendered_src_masks, smooth_ref_img
        )

        # source stage loss, synthesized depth -> reference images
        # ssim loss, reconstr loss for synthesized reference images
        # smooth loss for source synthesized images
        src_ssim_loss, src_reconstr_loss, src_smooth_loss = self.calc_src_loss_stage(
            syn_src_depths, gt_imgs, gt_masks, rendered_ref_imgs, rendered_ref_masks
        )

        # depth consistency loss
        # cur_masks = syn_src_depth_masks.to(dtype) * rendered_src_masks.to(dtype)
        # cur_masks = (gt_masks[:, 1:].to(dtype) * cur_masks) > 0
        cur_masks = syn_src_depth_masks * rendered_src_masks * gt_masks[:, 1:].to(torch.bool)
        depth_consistency = F.smooth_l1_loss(
            syn_src_depths[cur_masks], rendered_src_depths[cur_masks].detach()
        )
        depth_consistency = depth_consistency / cur_masks.size(1)

        # total loss
        total_stage_loss = depth_consistency_weight * depth_consistency + \
                           ssim_loss_weight * (ref_ssim_loss + src_ssim_loss) + \
                           smooth_loss_weight * (ref_smooth_loss + src_smooth_loss) + \
                           reconstr_loss_weight * (ref_reconstr_loss + src_reconstr_loss)
        
        # DEBUG
        self.debug_outputs[stage_key] = {
            "syn_src_depth": syn_src_depths[:, 0],
            "rendered_src_depth": rendered_src_depths[:, 0]
        }
                
        return total_stage_loss

    def forward(self, inputs, outputs, epoch, **kwargs):
        stage_weights = kwargs.get('stage_weights')
        ssim_loss_weight = kwargs.get('ssim_loss_weight')
        smooth_loss_weight = kwargs.get('smooth_loss_weight')
        reconstr_loss_weight = kwargs.get('reconstr_loss_weight')
        depth_consistency_weight = kwargs.get('depth_consistency_weight')

        ms_cameras = inputs['proj_matrices']
        gt_imgs, gt_masks = inputs['imgs'], inputs['masks']
        min_depth, max_depth = inputs['depth_values'][:, 0], inputs['depth_values'][:, -1]

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=gt_imgs.device, requires_grad=False)
        for (stage_outputs, stage_key) in [(outputs[k], k) for k in outputs.keys() if "stage" in k]:
            cameras = ms_cameras[stage_key]
            prob_volume = stage_outputs['prob_volume']
            depth_values = stage_outputs['depth_values']
            ref_depth = stage_outputs['depth'].unsqueeze(1)

            loss_stage = self.calc_loss_stage(
                gt_imgs, gt_masks, cameras, stage_key,
                ref_depth, depth_values, prob_volume, min_depth, max_depth,
                ssim_loss_weight, reconstr_loss_weight, smooth_loss_weight, depth_consistency_weight
            )

            stage_idx = int(stage_key.replace("stage", "")) - 1
            stage_weight = stage_weights[stage_idx]
            total_loss += stage_weight * loss_stage
        
        return total_loss


def mvs_loss():
    return UnSupLoss()    






