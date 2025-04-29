import math
import time
import torch
from torch import Tensor
import torch.nn as nn
from torchvision._internally_replaced_utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
import transformers
from clip import clip

# ----------------------------------------
# ResNetPrim Implementation
# ----------------------------------------
__all__ = ['ResNetPrim', 'resnet18_prim', 'resnet34_prim', 'resnet50_prim',
           'resnet101_prim', 'resnet152_prim']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    # ... other URLs ...
}


def safe_inverse(H: torch.Tensor) -> torch.Tensor:
    """
    Safely compute H^{-1}: try Cholesky + inverse, fallback to pinv.
    """
    # 1) Symmetrize
    H_sym = 0.5 * (H + H.transpose(-1, -2))
    # 2) Try Cholesky
    try:
        L = torch.linalg.cholesky(H_sym)
        return torch.cholesky_inverse(L)
    except RuntimeError:
        # 3) Fallback to pseudo-inverse
        return torch.linalg.pinv(H_sym, hermitian=True)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1,
            groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlockPrim(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1, base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, get_activations: bool=False):
        identity = x
        out1 = self.conv1(x)
        out2 = self.bn1(out1)
        out3 = self.relu(out2)
        out4 = self.conv2(out3)
        out5 = self.bn2(out4)
        if self.downsample is not None:
            identity = self.downsample(x)
        out6 = out5 + identity
        out7 = self.relu(out6)
        if get_activations:
            return out7, {
                'conv1': out1, 'bn1': out2, 'relu1': out3,
                'conv2': out4, 'bn2': out5, 'plus': out6, 'relu2': out7
            }
        return out7

class BottleneckPrim(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1, base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, get_activations: bool=False):
        identity = x
        out1 = self.conv1(x)
        out2 = self.bn1(out1)
        out3 = self.relu(out2)
        out4 = self.conv2(out3)
        out5 = self.bn2(out4)
        out6 = self.relu(out5)
        out7 = self.conv3(out6)
        out8 = self.bn3(out7)
        if self.downsample is not None:
            identity = self.downsample(x)
        out9 = out8 + identity
        out10 = self.relu(out9)
        if get_activations:
            return out10, {
                'conv1': out1, 'bn1': out2, 'relu1': out3,
                'conv2': out4, 'bn2': out5, 'relu2': out6,
                'conv3': out7, 'bn3': out8, 'plus': out9, 'relu3': out10
            }
        return out10

class ResNetPrim(nn.Module):
    def __init__(self, block: Type[Union[BasicBlockPrim, BottleneckPrim]],
                 layers: List[int], num_classes: int=1000,
                 zero_init_residual: bool=False,
                 groups: int=1, width_per_group: int=64,
                 replace_stride_with_dilation: Optional[List[bool]]=None,
                 norm_layer: Optional[Callable[..., nn.Module]]=None) -> None:
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckPrim):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockPrim):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

# Factory for ResNetPrim backbones
def resnet50_prim(pretrained: bool=False, **kwargs) -> ResNetPrim:
    model = ResNetPrim(BottleneckPrim, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        model.load_state_dict(state)
    return model

# ----------------------------------------
# SparseCLIP Implementation
# ----------------------------------------

class SparseGPT:
    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(layer, transformers.Conv1D):
            W = W.t()
        self.rows, self.columns = W.shape
        self.nsamples = 0
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)

    def add_batch(self, inp: Tensor, out: Tensor, blocksize: int = 1024):
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                kernel_size=self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            patches = unfold(inp)
            inp_flat = patches.permute(1, 0, 2).reshape(self.columns, -1)
        else:
            inp2 = inp.reshape(-1, inp.shape[-1]) if inp.dim() > 2 else inp
            inp_flat = inp2.t()
        n_new = inp_flat.shape[1]
        self.H *= (self.nsamples / (self.nsamples + n_new))
        self.nsamples += n_new
        scaled = math.sqrt(2.0 / self.nsamples) * inp_flat.float()
        self.H += scaled.matmul(scaled.t())

    def fasterprune(self, sparsity, percdamp=1e-2): # Removed unused blocksize
        W = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d): W = W.flatten(1)
        # Remove Conv1D check if not needed for your CLIP models
        if isinstance(self.layer, transformers.Conv1D): W = W.t()

        # Damped Hessian (H = H + lambda * diag(H) * I) - Check if this damping is optimal
        H_damped = self.H + percdamp * torch.mean(torch.diagonal(self.H)) * torch.eye(self.columns, device=self.dev)

        # Inverse via safe method
        Hinv = safe_inverse(H_damped)

        # --- CRITICAL CHANGE: Corrected Score Calculation ---
        # Using OBS-like criterion: Error increase approx W^2 / (2 * diag(H^-1))
        # We ignore the factor of 2 as it doesn't affect ranking
        Hinv_diag = torch.diagonal(Hinv)
        epsilon = 1e-9 # For numerical stability
        score = W.pow(2) / (Hinv_diag.unsqueeze(0) + epsilon)
        # ----------------------------------------------------

        # Ensure score shape matches W shape if needed (might depend on W manipulations)
        if score.shape != W.shape:
            # This might happen if W was transposed, ensure score aligns
            # Example: if W is (rows, columns), score might need broadcasting or alignment
            # Needs careful checking based on layer type and W manipulation
            print(f"Warning: Score shape {score.shape} mismatch with W shape {W.shape} in layer {type(self.layer)}")
            # Adjust score shape if necessary, e.g., ensure it's (rows, columns) matching W
            # This depends heavily on how W was flattened/transposed earlier.
            # Assuming score calculation yields (1, columns), needs broadcasting to W's rows.
            # If W is (rows, columns), Hinv_diag is (columns), score becomes (rows, columns) via broadcasting. This should be okay.

        # Magnitude-based mask using the corrected score
        # Keep weights with scores *below* the threshold (low error impact)
        thresh = torch.quantile(score.flatten(), sparsity) # Flatten score to compute quantile correctly
        mask = score > thresh # Keep weights with score > threshold (higher importance relative to error increase)
                            # Or use 'score <= thresh' to prune low-impact weights (more common for OBS)
                            # Let's stick to pruning low-impact weights:
        mask = score <= thresh

        # Apply mask
        W_pruned = W * (~mask) # Apply inverse mask to keep important weights

        # Reshape back (ensure Wp assignment handles Conv1D transpose correctly)
        if isinstance(self.layer, transformers.Conv1D):
            Wp = W_pruned.t()
        else:
            Wp = W_pruned

        self.layer.weight.data.copy_(Wp.reshape(self.layer.weight.shape))

        # Free memory
        del self.H
        del H_damped
        del Hinv
        torch.cuda.empty_cache() # Good practice

class SparseCLIP:
    """
    Integrates SparseGPT pruning into CLIP with support for ResNetPrim backbones.
    """
    def __init__(self, model, visual_backbone: str='default'):
        self.model = model
        self.dev = next(model.parameters()).device
        self.prunable_layers = {}
        self.sparse_layers = {}
        # Collect layers from visual encoder
        if visual_backbone == 'resnetprim':
            visual = self.model.visual
        else:
            visual = self.model.visual
        for name, module in visual.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight.requires_grad:
                self.prunable_layers[f"visual.{name}"] = module
        # Collect from text encoder
        for name, module in self.model.transformer.named_modules():
            if isinstance(module, (nn.Linear, transformers.Conv1D)) and module.weight.requires_grad:
                self.prunable_layers[f"text.{name}"] = module
        # Init SparseGPT
        for name, layer in self.prunable_layers.items():
            self.sparse_layers[name] = SparseGPT(layer)

    def add_batch(self, images, texts):
        hooks = []
        inputs, outputs = {}, {}
        def hook_fn(key):
            def fn(mod, inp, out):
                inputs[key] = inp[0] if isinstance(inp, tuple) else inp
                outputs[key] = out
            return fn
        for name, layer in self.prunable_layers.items():
            hooks.append(layer.register_forward_hook(hook_fn(name)))
        # forward
        _ = self.model.encode_image(images)
        _ = self.model.encode_text(texts)
        # remove hooks
        for h in hooks: h.remove()
        # collect
        for name, s in self.sparse_layers.items():
            if name in inputs and name in outputs:
                s.add_batch(inputs[name].detach(), outputs[name].detach())

    def prune(self, sparsity: float):
        for name, s in self.sparse_layers.items():
            print(f"Pruning layer: {name}")
            s.fasterprune(sparsity)
        torch.cuda.empty_cache()

    def eval(self, dataset, batch_size=32):
        self.model.eval()
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, txts in loader:
                imgs, txts = imgs.to(self.dev), clip.tokenize(txts).to(self.dev)
                imf = self.model.encode_image(imgs)
                tf  = self.model.encode_text(txts)
                imf = imf / imf.norm(dim=1, keepdim=True)
                tf  = tf / tf.norm(dim=1, keepdim=True)
                sims = (100*imf @ tf.t())
                vals, inds = sims.topk(1)
                correct += sum(int(i==j) for i,j in zip(inds.flatten(), range(len(inds))))
                total += len(inds)
        return correct/total

# Helper to load and apply

def apply_sparseclip(model_name: str="ViT-B/32", dataset=None, sparsity: float=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- CRITICAL CHANGE: Load Model Correctly ---
    # Load the desired CLIP model directly.
    # If using ResNet, use "RN50", "RN101", etc.
    print(f"Loading CLIP model: {model_name}")
    model, preprocess = clip.load(model_name, device=device)

    # Remove the ResNetPrim swapping logic unless you have a specific,
    # well-justified reason and implement weight loading correctly (Option B above).
    # --------------------------------------------

    if dataset is None:
        print("Warning: No calibration dataset provided. Pruning will not be performed.")
        return model, preprocess # Return original model if no dataset

    print("Initializing SparseCLIP...")
    # Pass 'default' or remove backbone argument if ResNetPrim isn't used
    sparse = SparseCLIP(model)

    print("Starting calibration (calculating Hessians)...")
    # Consider using a subset of the dataset for calibration if it's large
    # Also consider a larger batch size if memory allows
    calibration_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True) # Shuffle for better Hessian estimate
    num_batches_for_calibration = min(100, len(calibration_loader)) # Limit calibration steps? Example: use 100 batches

    count = 0
    start_time = time.time()
    for imgs, txts in calibration_loader:
        # Ensure preprocessing is applied if the dataset doesn't provide tensors directly
        # Assuming dataset yields PIL images and strings:
        # processed_imgs = torch.stack([preprocess(img) for img in imgs]).to(device)
        # tokenized_txts = clip.tokenize(list(txts)).to(device)

        # If dataset already provides tensors:
        processed_imgs = imgs.to(device)
        # Assuming txts are strings that need tokenizing
        if isinstance(txts, (list, tuple)) and isinstance(txts[0], str):
             tokenized_txts = clip.tokenize(txts).to(device)
        else: # Assume txts are already tokenized tensors
             tokenized_txts = txts.to(device)


        sparse.add_batch(processed_imgs, tokenized_txts)
        count += 1
        if count >= num_batches_for_calibration:
             print(f"Completed calibration using {count} batches.")
             break
    print(f"Calibration finished in {time.time() - start_time:.2f} seconds.")


    print(f"Pruning model with sparsity: {sparsity}")
    start_time = time.time()
    sparse.prune(sparsity)
    print(f"Pruning finished in {time.time() - start_time:.2f} seconds.")


    print("Evaluating pruned model...")
    # Ensure evaluation uses the correct preprocessor and tokenization
    # The eval function might need adjustment depending on the dataset format
    acc = sparse.eval(dataset) # Pass the *original* dataset instance
    print(f"Accuracy after pruning: {acc:.4f}")

    return model, preprocess # Return the pruned model
