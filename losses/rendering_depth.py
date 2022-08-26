import torch


@torch.no_grad()
def _get_warping_grid(hypos, ref_cam, src_cam):
    """
    homography matrix from source to reference view.

    Args:
        hypos: reference camera coords depth hypos, [B, D, H, W]
        ref_proj: reference camera project matrix, [B, 4, 4]
        src_proj: source camera project matrix, [B, 4, 4]
    Returns:
        grid: homography matrix, [B, 2, D, H, W]
    """
    device = hypos.device
    batch, num_hypos, height, width = hypos.shape

    src_proj = src_cam[:, 0].clone()
    src_proj[:, :3, :4] = torch.matmul(src_cam[:, 1, :3, :3], src_cam[:, 0, :3, :4])
    ref_proj = ref_cam[:, 0].clone()
    ref_proj[:, :3, :4] = torch.matmul(ref_cam[:, 1, :3, :3], ref_cam[:, 0, :3, :4])

    proj = torch.matmul(src_proj, torch.inverse(ref_proj))
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
    rot_depth_xyz = rot_xyz * hypos.view(batch, 1, num_hypos, -1)  # [B, 3, D, H*W]
    proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, D, H*W]
    proj_xyz[:, 2:3][proj_xyz[:, 2:3] == 0] += 0.00001  # NAN BUG, not on dtu, but on blended
    proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, D, H*W]

    grid = proj_xy.view(batch, 2, num_hypos, height, width)  # [B, 2, D, H, W]
    depth = proj_xyz[:, 2].view(batch, num_hypos, height, width)  # [B, D, H, W]

    return grid, depth


def __splat__(values, coords):
    """
    splatting depth or weight.

    Args:
        values: depth or weights, [B, D, H, W]
        coords: warp grid from reference to source, [B, 2, D, H, W]
    Returns:
        splatted: splatted values, [B, 1, H, W]
    """
    device, dtype = values.device, values.dtype
    batch, num_hypos, height, width = values.shape

    values = values.view(-1, 1, height, width)
    splatted = torch.zeros_like(values)
    coords = coords.permute(0, 2, 1, 3, 4).reshape(-1, 2, height, width)

    uvs = coords
    u = uvs[:, 0, :, :].unsqueeze(1)
    v = uvs[:, 1, :, :].unsqueeze(1)

    u0 = torch.floor(u)
    u1 = u0 + 1
    v0 = torch.floor(v)
    v1 = v0 + 1

    u0_safe = torch.clamp(u0, 0.0, width - 1)
    u1_safe = torch.clamp(u1, 0.0, width - 1)
    v0_safe = torch.clamp(v0, 0.0, height - 1)
    v1_safe = torch.clamp(v1, 0.0, height - 1)

    u0_w = (u1 - u) * (u0 == u0_safe).detach().to(dtype)
    u1_w = (u - u0) * (u1 == u1_safe).detach().to(dtype)
    v0_w = (v1 - v) * (v0 == v0_safe).detach().to(dtype)
    v1_w = (v - v0) * (v1 == v1_safe).detach().to(dtype)

    top_left_w = u0_w * v0_w
    top_right_w = u1_w * v0_w
    bottom_left_w = u0_w * v1_w
    bottom_right_w = u1_w * v1_w

    threshold = 1e-3  # weight threshold
    top_left_w *= (top_left_w >= threshold).detach().to(dtype)
    top_right_w *= (top_right_w >= threshold).detach().to(dtype)
    bottom_left_w *= (bottom_left_w >= threshold).detach().to(dtype)
    bottom_right_w *= (bottom_right_w >= threshold).detach().to(dtype)

    top_left_values = values * top_left_w
    top_right_values = values * top_right_w
    bottom_left_values = values * bottom_left_w
    bottom_right_values = values * bottom_right_w

    new_batch = batch * num_hypos
    top_left_values = top_left_values.reshape(new_batch, -1)
    top_right_values = top_right_values.reshape(new_batch, -1)
    bottom_left_values = bottom_left_values.reshape(new_batch, -1)
    bottom_right_values = bottom_right_values.reshape(new_batch, -1)

    top_left_indices = (u0_safe + v0_safe * width).reshape(new_batch, -1).long()
    top_right_indices = (u1_safe + v0_safe * width).reshape(new_batch, -1).long()
    bottom_left_indices = (u0_safe + v1_safe * width).reshape(new_batch, -1).long()
    bottom_right_indices = (u1_safe + v1_safe * width).reshape(new_batch, -1).long()

    splatted = splatted.reshape(new_batch, -1)
    splatted.scatter_add_(1, top_left_indices, top_left_values)
    splatted.scatter_add_(1, top_right_indices, top_right_values)
    splatted.scatter_add_(1, bottom_left_indices, bottom_left_values)
    splatted.scatter_add_(1, bottom_right_indices, bottom_right_values)

    splatted = splatted.reshape(batch, num_hypos, height, width)

    return torch.sum(splatted, dim=1, keepdims=True)


def __weighted_average_splat__(depth, weights, epsilon=1e-8):
    zero_weights = (weights <= epsilon).detach().type(depth.dtype)
    return depth / (weights + epsilon * zero_weights)


def get_render_srcdepth(ref_hypos, ref_cost, cameras, min_depth, max_depth):
    """
    rendering source depths, using the reference
    cost volume and depth hypotheses. The key idea is intersection
    points of ray and plane.

    Args:
        ref_hypos: depth hypothesis of reference view, [B, D, H, W]
        ref_cost: cost volume of the reference view, [B, D, H, W]
        cameras: reference and source images camera matrix, [B, V, 2, 4, 4]
        srcs_depth_range: image depth ranges, [B, V-1, 2]
        min_depth: reference min depth [B, ]
        max_depth: reference max depth [B, ]
    Returns:
        src_depths: rendering source image depths, [B, V-1, H, W]
        src_masks: rendering source image masks, [B, V-1, H, W]
    """
    batch, _, height, width = ref_hypos.shape

    cameras = torch.unbind(cameras, 1)
    ref_cam, src_cams = cameras[0], cameras[1:]

    min_depth = min_depth.view(batch, 1, 1, 1).repeat(1, 1, height, width)
    max_depth = max_depth.view(batch, 1, 1, 1).repeat(1, 1, height, width)

    src_depths, src_masks = [], []
    for idx, src_cam in enumerate(src_cams):
        warping_grid, warping_depth = _get_warping_grid(ref_hypos, ref_cam, src_cam)
        splat_depth = __splat__(ref_cost * warping_depth, warping_grid)
        splat_weight = __splat__(ref_cost, warping_grid)
        src_depth = __weighted_average_splat__(splat_depth, splat_weight)
        src_mask = (splat_weight > 1e-3).detach()

        src_depth += torch.where(min_depth > src_depth,
                min_depth - src_depth, torch.zeros_like(src_depth))
        src_depth -= torch.where(max_depth < src_depth,
                src_depth - max_depth, torch.zeros_like(src_depth))
        
        src_masks.append(src_mask)
        src_depths.append(src_depth)

    src_masks = torch.cat(src_masks, dim=1)
    src_depths = torch.cat(src_depths, dim=1)

    return src_depths, src_masks