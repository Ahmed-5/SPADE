import math
import time

import torch
import torch.nn as nn
import transformers
from clip import clip

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseCLIP:
    def __init__(self, model):
        """
        Initialize SparseCLIP with a CLIP model
        
        Args:
            model: A CLIP model (containing visual and text encoders)
        """
        self.model = model
        self.dev = next(model.parameters()).device
        self.prunable_layers = self._get_prunable_layers()
        self.sparse_layers = {}
        
        # Initialize SparseGPT for each prunable layer
        for name, layer in self.prunable_layers.items():
            self.sparse_layers[name] = SparseGPT(layer)
            
    def _get_prunable_layers(self):
        """
        Get all prunable layers (Linear, Conv) from the CLIP model
        
        Returns:
            Dict of prunable layers with their names
        """
        prunable_layers = {}
        
        # Handle visual encoder
        for name, module in self.model.visual.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and module.weight.requires_grad:
                prunable_layers[f"visual.{name}"] = module
                
        # Handle text encoder
        for name, module in self.model.transformer.named_modules():
            if isinstance(module, (nn.Linear, transformers.Conv1D)) and module.weight.requires_grad:
                prunable_layers[f"text.{name}"] = module
                
        return prunable_layers
    
    def add_batch(self, images, texts, blocksize=1024):
        """
        Process a batch through the model and collect statistics for pruning
        
        Args:
            images: Batch of images
            texts: Batch of text tokens
            blocksize: Block size for processing
        """
        # Store original forward hooks to restore later
        original_hooks = {}
        hook_handles = []
        layer_inputs = {}
        layer_outputs = {}
        
        # Register forward hooks to capture inputs and outputs
        def hook_fn(name):
            def _hook(module, inp, out):
                if isinstance(inp, tuple):
                    inp = inp[0]
                layer_inputs[name] = inp.detach()
                layer_outputs[name] = out.detach()
            return _hook
        
        for name, layer in self.prunable_layers.items():
            handle = layer.register_forward_hook(hook_fn(name))
            hook_handles.append(handle)
            
        # Forward pass
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(texts)
        
        # Remove hooks
        for handle in hook_handles:
            handle.remove()
            
        # Add batch data to each sparse layer
        for name, sparse_layer in self.sparse_layers.items():
            if name in layer_inputs and name in layer_outputs:
                sparse_layer.add_batch(layer_inputs[name], layer_outputs[name], blocksize)
                
    def prune(self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01):
        """
        Prune the CLIP model with the specified sparsity
        
        Args:
            sparsity: Target sparsity level (0.0 to 1.0)
            prunen, prunem: Advanced pruning parameters
            blocksize: Block size for pruning
            percdamp: Damping factor for numerical stability
        """
        print(f"Pruning CLIP model to {sparsity*100:.1f}% sparsity")
        
        # Prune each layer
        for name, sparse_layer in self.sparse_layers.items():
            print(f"Pruning layer: {name}")
            sparse_layer.fasterprune(sparsity, prunen, prunem, blocksize, percdamp)
            sparse_layer.free()  # Free memory
            
        # Clear GPU memory
        torch.cuda.empty_cache()
        
    def eval_model(self, eval_dataset, batch_size=32):
        """
        Evaluate the pruned CLIP model on a dataset
        
        Args:
            eval_dataset: Dataset for evaluation
            batch_size: Batch size for evaluation
        
        Returns:
            Accuracy metrics
        """
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, texts in dataloader:
                images = images.to(self.dev)
                texts = clip.tokenize(texts).to(self.dev)
                
                # Get features
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(texts)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T)
                
                # Get predictions
                values, indices = similarity.topk(1)
                
                # Count correct predictions (diagonal matches)
                for i in range(len(indices)):
                    if indices[i] == i:
                        correct += 1
                total += len(indices)
        
        return correct / total


class SparseGPT:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
            
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
            
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.nsamples = 0

        if isinstance(self.layer, nn.Conv2d) and self.layer.groups > 1:
            self.H = torch.zeros((W.shape[0], self.columns, self.columns), device=self.dev)
        else:
            self.H = torch.zeros((self.columns, self.columns), device=self.dev)
            
    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) >= 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
            
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                    self.layer.kernel_size,
                    dilation=self.layer.dilation,
                    padding=self.layer.padding,
                    stride=self.layer.stride
                )
            channels=inp.shape[1]
            inp = unfold(inp)
            
            if self.layer.groups == 1:
                inp = inp.permute([1, 0, 2])
                inp = inp.flatten(1)
            else:
                inp = inp.reshape((inp.shape[0], channels, inp.shape[1]//channels, inp.shape[2]))
                inp = inp.permute([2, 0, 1, 3])

        if isinstance(self.layer, nn.Conv2d) and self.layer.groups > 1:
            inp = inp.flatten(2)
            
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        
    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()


# Example usage
def apply_sparseclip(model_name="ViT-B/32", dataset=None, sparsity=0.5):
    """
    Apply SparseCLIP to a CLIP model
    
    Args:
        model_name: CLIP model name
        dataset: Dataset for calibration and evaluation
        sparsity: Target sparsity level (0.0 to 1.0)
        
    Returns:
        Pruned CLIP model
    """
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    
    # Create SparseCLIP instance
    sparse_clip = SparseCLIP(model)
    
    # If dataset is provided
    if dataset is not None:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        # Collect statistics from calibration data
        print("Collecting statistics for pruning...")
        for images, texts in dataloader:
            images = images.to(device)
            texts = clip.tokenize(texts).to(device)
            sparse_clip.add_batch(images, texts)
            
        # Prune the model
        sparse_clip.prune(sparsity)
        
        # Evaluate the pruned model
        accuracy = sparse_clip.eval_model(dataset)
        print(f"Pruned model accuracy: {accuracy:.4f}")
    
    return model