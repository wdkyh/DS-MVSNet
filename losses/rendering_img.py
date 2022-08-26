import torch


def __splat__(values, coords, splatted):
    b, c, h, w = splatted.size()
    uvs = coords
    u = uvs[:, 0, :, :].unsqueeze(1)
    v = uvs[:, 1, :, :].unsqueeze(1)
    
    u0 = torch.floor(u)
    u1 = u0 + 1
    v0 = torch.floor(v)
    v1 = v0 + 1
    u0_safe = torch.clamp(u0, 0.0, w-1)
    v0_safe = torch.clamp(v0, 0.0, h-1)
    u1_safe = torch.clamp(u1, 0.0, w-1)
    v1_safe = torch.clamp(v1, 0.0, h-1)

    u0_w = (u1 - u) * (u0 == u0_safe).detach().type(values.dtype)
    u1_w = (u - u0) * (u1 == u1_safe).detach().type(values.dtype)
    v0_w = (v1 - v) * (v0 == v0_safe).detach().type(values.dtype)
    v1_w = (v - v0) * (v1 == v1_safe).detach().type(values.dtype)

    top_left_w = u0_w * v0_w
    top_right_w = u1_w * v0_w
    bottom_left_w = u0_w * v1_w
    bottom_right_w = u1_w * v1_w

    weight_threshold = 1e-3
    top_left_w *= (top_left_w >= weight_threshold).detach().type(values.dtype)
    top_right_w *= (top_right_w >= weight_threshold).detach().type(values.dtype)
    bottom_left_w *= (bottom_left_w >= weight_threshold).detach().type(values.dtype)
    bottom_right_w *= (bottom_right_w >= weight_threshold).detach().type(values.dtype)
    for channel in range(c):
        top_left_values = values[:, channel, :, :].unsqueeze(1) * top_left_w
        top_right_values = values[:, channel, :, :].unsqueeze(1) * top_right_w
        bottom_left_values = values[:, channel, :, :].unsqueeze(1) * bottom_left_w
        bottom_right_values = values[:, channel, :, :].unsqueeze(1) * bottom_right_w

        top_left_values = top_left_values.reshape(b, -1)
        top_right_values = top_right_values.reshape(b, -1)
        bottom_left_values = bottom_left_values.reshape(b, -1)
        bottom_right_values = bottom_right_values.reshape(b, -1)

        top_left_indices = (u0_safe + v0_safe * w).reshape(b, -1).type(torch.int64)
        top_right_indices = (u1_safe + v0_safe * w).reshape(b, -1).type(torch.int64)
        bottom_left_indices = (u0_safe + v1_safe * w).reshape(b, -1).type(torch.int64)
        bottom_right_indices = (u1_safe + v1_safe * w).reshape(b, -1).type(torch.int64)
        
        splatted_channel = splatted[:, channel, :, :].unsqueeze(1)
        splatted_channel = splatted_channel.reshape(b, -1)
        splatted_channel.scatter_add_(1, top_left_indices, top_left_values)
        splatted_channel.scatter_add_(1, top_right_indices, top_right_values)
        splatted_channel.scatter_add_(1, bottom_left_indices, bottom_left_values)
        splatted_channel.scatter_add_(1, bottom_right_indices, bottom_right_values)
    splatted = splatted.reshape(b, c, h, w)

def __weighted_average_splat__(depth, weights, epsilon=1e-8):
    zero_weights = (weights <= epsilon).detach().type(depth.dtype)
    return depth / (weights + epsilon * zero_weights)

def __depth_distance_weights__(depth, max_depth=20.0):
    weights = 1.0 / torch.exp(2 * depth / max_depth)
    return weights

def render(img, depth, warping_depth, coords, max_depth=20.0, depth_mask=None):
    splatted_img = torch.zeros_like(img)
    splatted_wgts = torch.zeros_like(warping_depth)        
    weights = __depth_distance_weights__(warping_depth, max_depth=max_depth)
    if depth_mask is not None: weights = weights * depth_mask.to(torch.float32)
    __splat__(weights, coords, splatted_wgts)
    __splat__(img * weights, coords, splatted_img)
    recon_img = __weighted_average_splat__(splatted_img, splatted_wgts)
    if depth is not None:
        splatted_depth = torch.zeros_like(depth)
        __splat__(depth * weights, coords, splatted_depth)
        recon_depth = __weighted_average_splat__(splatted_depth, splatted_wgts)
    else:
        recon_depth = None
    mask = (splatted_wgts > 1e-3).detach()

    return recon_img, recon_depth, mask


def get_warping_grid(depth, cam_a, cam_b):
    """
    homography matrix from a to b view.
    Args:
        depth: a camera coords depth, [B, 1, H, W]
        cam_a: a camera project matrix, [B, 4, 4]
        cam_b: b camera project matrix, [B, 4, 4]
    Returns:
        grid: homography matrix a to b, [B, 2, D, H, W]
        depth: b camera coords depth, [B, 1, H, W]
    """
    device = depth.device
    batch, num_hypos, height, width = depth.shape
    
    proj_b = cam_b[:, 0].clone()
    proj_b[:, :3, :4] = torch.matmul(cam_b[:, 1, :3, :3], cam_b[:, 0, :3, :4])
    proj_a = cam_a[:, 0].clone()
    proj_a[:, :3, :4] = torch.matmul(cam_a[:, 1, :3, :3], cam_a[:, 0, :3, :4])

    proj = torch.matmul(proj_b, torch.inverse(proj_a))
    rot = proj[:, :3, :3]  # [B, 3, 3]
    trans = proj[:, :3, 3:4]  # [B, 3, 1]

    y, x = torch.meshgrid(
        [
            torch.arange(0, height, dtype=torch.float32, device=device),
            torch.arange(0, width, dtype=torch.float32, device=device),
        ]
    )
    y, x = y.contiguous(), x.contiguous()

    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
    rot_xyz = torch.matmul(rot, xyz).unsqueeze(2).repeat(1, 1, num_hypos, 1)
    rot_depth_xyz = rot_xyz * depth.view(batch, 1, num_hypos, -1)
    proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, 1, H*W]
    # NAN BUG, not on dtu, but on blended
    proj_xyz[:, 2:3] = proj_xyz[:, 2:3] + 1e-6 * (proj_xyz[:, 2:3] == 0).to(torch.float32)

    proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, 1, H*W]

    grid = proj_xy.view(batch, 2, num_hypos, height, width)  # [B, 2, 1, H, W]

    depth = proj_xyz[:, 2].view(batch, num_hypos, height, width)  # [B, 1, H, W]
    return grid.squeeze(2), depth


def get_rendered_src(ref_img, ref_depth, cameras, max_depth):
    cameras = torch.unbind(cameras, 1)
    ref_cam, src_cams = cameras[0], cameras[1:]

    src_imgs, src_depths, src_masks = [], [], []
    for src_cam in src_cams:
        warping_grid, warping_depth = get_warping_grid(ref_depth, ref_cam, src_cam)
        src_img, src_depth, src_mask = render(
            ref_img, ref_depth, warping_depth, warping_grid, max_depth=max_depth
        )
        src_imgs.append(src_img)
        src_masks.append(src_mask)
        src_depths.append(src_depth)

    src_masks = torch.cat(src_masks, dim=1)
    src_imgs = torch.stack(src_imgs, dim=1)
    src_depths = torch.cat(src_depths, dim=1)
    return src_imgs, src_depths, src_masks  # [B, V-1, C, H, W], [B, V-1, H, W], [B, V-1, H, W]


def get_rendered_ref(imgs, src_depths, depth_masks, cameras, max_depth):
    cameras = torch.unbind(cameras, 1)
    ref_cam, src_cams = cameras[0], cameras[1:]

    ref_imgs, ref_masks = [], []
    for idx, src_cam in enumerate(src_cams):
        warping_grid, warping_depth = get_warping_grid(
            src_depths[:, idx].unsqueeze(1), src_cam, ref_cam
        )
        ref_img, _, ref_mask = render(
            imgs[:, idx], None,
            warping_depth, warping_grid, max_depth, depth_masks[:, idx:idx+1]
        )
        ref_imgs.append(ref_img)
        ref_masks.append(ref_mask)
    
    ref_imgs = torch.stack(ref_imgs, dim=1)
    ref_masks = torch.cat(ref_masks, dim=1)
    # [B, V-1, C, H, W], [B, V-1, H, W], [B, V-1, H, W]
    return ref_imgs, ref_masks