import torch
import torch.nn.functional as F



def _gradient_x(img: torch.Tensor) -> torch.Tensor:
    if len(img.shape) != 4:
        raise AssertionError(img.shape)
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def _gradient_y(img: torch.Tensor) -> torch.Tensor:
    if len(img.shape) != 4:
        raise AssertionError(img.shape)
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def depth_smoothness_loss(depth: torch.Tensor, image: torch.Tensor, reduction: str = 'none') -> torch.Tensor:
    """
    Args:
        depth: tensor with the depth with shape :math:`(N, 1, H, W)`.
        image: tensor with the input image with shape :math:`(N, 3, H, W)`.
    Return:
        a scalar with the computed loss.
    """
    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"Input depth type is not a torch.Tensor. Got {type(depth)}")

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input image type is not a torch.Tensor. Got {type(image)}")

    if not len(depth.shape) == 4:
        raise ValueError(f"Invalid depth shape, we expect BxCxHxW. Got: {depth.shape}")

    if not len(image.shape) == 4:
        raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")

    if not depth.shape[-2:] == image.shape[-2:]:
        raise ValueError(f"depth and image shapes must be the same. Got: {depth.shape} and {image.shape}")

    if not depth.device == image.device:
        raise ValueError(f"depth and image must be in the same device. Got: {depth.device} and {image.device}")

    if not depth.dtype == image.dtype:
        raise ValueError(f"depth and image must be in the same dtype. Got: {depth.dtype} and {image.dtype}")

    # compute the gradients
    depth_dx: torch.Tensor = _gradient_x(depth)
    depth_dy: torch.Tensor = _gradient_y(depth)
    image_dx: torch.Tensor = _gradient_x(image)
    image_dy: torch.Tensor = _gradient_y(image)

    # compute image weights
    weights_x: torch.Tensor = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y: torch.Tensor = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    # apply image weights to depth
    smoothness_x: torch.Tensor = torch.abs(depth_dx * weights_x)
    smoothness_y: torch.Tensor = torch.abs(depth_dy * weights_y)

    smoothness_x = F.pad(smoothness_x, [0, 1, 0, 0], mode='replicate')
    smoothness_y = F.pad(smoothness_y, [0, 0, 0, 1], mode='replicate')

    if reduction == "mean":
        return torch.mean(smoothness_x + smoothness_y)
    elif reduction == "sum":
        return torch.sum(smoothness_x + smoothness_y)
    elif reduction == "none":
        return smoothness_x + smoothness_y


def inverse_depth_smoothness_loss(depth, image, reduction='none'):
    """
    Args:
        depth: tensor with the depth with shape :math:`(N, 1, H, W)`.
        image: tensor with the input image with shape :math:`(N, 3, H, W)`.
    Return:
        a scalar with the computed loss.
    """
    mask = depth <= 0
    depth_min = depth[depth > 0].min()
    depth = torch.clamp(depth, min=depth_min.item())
    idepth = 1.0 / depth
    return depth_smoothness_loss(idepth, image, reduction)