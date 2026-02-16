import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.transforms.functional as torchvision_F
from matplotlib import pyplot as plt
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import yaml
import os
import copy
from typing import Dict

class TwoViewDataset(Dataset):
    def __init__(self, base_dataset, teacher_tf, student_tf, is_train = True):
        self.base = base_dataset
        self.teacher_tf = teacher_tf
        self.student_tf = student_tf
        self.is_train = is_train

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]   # img is PIL image
        x_s = self.student_tf(img)
        if self.is_train == True : 
            x_t = self.teacher_tf(img)
        else : 
            return x_s, y
        return x_t, x_s, y


def load_CIFAR10_distill_dataset(
    batch_size: int = 128,
    calibration_batch_size: int = 1024,
    data_path: str = './data',
    teacher_train_transform=None,
    teacher_test_transform=None,
    student_train_transform=None,
    student_test_transform=None,
):
    """
    Loader for teacher-student distillation.
    Returns:
        train_loader, test_loader, calibration_loader
    """

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    num_workers = os.cpu_count()

    # Default student transforms (CIFAR-style)
    if student_train_transform is None:
        student_train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    if student_test_transform is None:
        student_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # Default teacher transforms = same as student if not provided
    if teacher_train_transform is None:
        teacher_train_transform = student_train_transform
    if teacher_test_transform is None:
        teacher_test_transform = student_test_transform

    # Base datasets (NO transforms)
    base_train = dset.CIFAR10(
        data_path,
        train=True,
        transform=None,
        download=True,
    )

    base_test = dset.CIFAR10(
        data_path,
        train=False,
        transform=None,
        download=True,
    )

    # Wrap into two-view datasets
    train_data = TwoViewDataset(
        base_train,
        teacher_tf=teacher_train_transform,
        student_tf=student_train_transform,
    )

    calibration_data = TwoViewDataset(
        base_train,
        teacher_tf=teacher_test_transform,  
        student_tf=student_test_transform,
        is_train=False
    )

    test_data = TwoViewDataset(
        base_test,
        teacher_tf=teacher_test_transform,   # usually unused
        student_tf=student_test_transform,
        is_train=False
    )

    num_train = len(base_train)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    train_idx = indices[calibration_batch_size:]
    calibration_idx = indices[:calibration_batch_size]

    train_sampler = SubsetRandomSampler(train_idx)
    calibration_sampler = SubsetRandomSampler(calibration_idx)

    # Standard loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    calibration_loader = DataLoader(
        calibration_data,
        batch_size=calibration_batch_size,
        sampler=calibration_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, calibration_loader

def load_CIFAR10_dataset(batch_size: int = 128, calibration_batch_size: int = 1024,
                         data_path: str = './data'):
    """
    download and loading the data loaders
    Args:
        batch_size: batch size for train and test loader
        calibration_batch_size: size of the calibration batch
        data_path: directory to save data

    Returns:
        train_loader, test_loader, calibration_loader
    """
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    num_workers = os.cpu_count()

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_data = dset.CIFAR10(data_path,
                              train=True,
                              transform=train_transform,
                              download=True)
    test_data = dset.CIFAR10(data_path,
                             train=False,
                             transform=test_transform,
                             download=True)
    calibration_data = dset.CIFAR10(data_path,
                                    train=True,
                                    transform=test_transform,
                                    download=False)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    train_idx, calibration_idx = indices[calibration_batch_size:], indices[:calibration_batch_size]
    train_sampler = SubsetRandomSampler(train_idx)
    calibration_sampler = SubsetRandomSampler(calibration_idx)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)
    calibration_loader = DataLoader(
        calibration_data,
        batch_size=calibration_batch_size,
        sampler=calibration_sampler,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader, test_loader, calibration_loader


def show_samples(test_data):
    """
    plot 4 samples of each classes in CIFAR10
    Args:
        test_data:

    Returns:

    """
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    samples = [[] for _ in range(10)]
    for image, label in test_data:
        if len(samples[label]) < 4:
            samples[label].append(image)

    fig, axes = plt.subplots(4, 10, squeeze=False, figsize=(10 * 3, 4 * 3))
    for i in range(10):
        for j in range(4):
            img = samples[i][j].detach()
            for c in range(img.shape[0]):
                img[c] = img[c] * std[c] + mean[c]
            img = torchvision_F.to_pil_image(img)

            axes[j, i].imshow(np.asarray(img))
            axes[j, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axes[j, i].set_title(test_data.classes[i])


def train(
        model: nn.Module,
        dataflow: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        device: torch.device = torch.device("cuda")
) -> None:
    model.train()

    for inputs, targets in dataflow:
        # Move the data from CPU to GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Reset the gradients (from the last iteration)
        optimizer.zero_grad()

        # Forward inference
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward propagation
        loss.backward()

        # Update optimizer and LR scheduler
        optimizer.step()
        if scheduler is not None:
            scheduler.step()


@torch.inference_mode()
def evaluate(
        model: nn.Module,
        dataflow: DataLoader,
        device: torch.device = torch.device("cuda")
) -> float:
    model.eval()

    num_samples = 0
    num_correct = 0

    for batch in dataflow:
        # Handle both (x, y) and (x_t, x_s, y)
        if len(batch) == 2:
            inputs, targets = batch
        elif len(batch) == 3:
            _, inputs, targets = batch   # use student view
        else:
            raise ValueError("Unexpected batch format")

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        preds = outputs.argmax(dim=1)

        num_samples += targets.size(0)
        num_correct += (preds == targets).sum()

    return (num_correct / num_samples * 100).item()

def fit(model: nn.Module, num_epochs: int, train_loader: DataLoader, test_loader: DataLoader,
        criterion: nn.Module, optimizer: Optimizer, scheduler: LambdaLR, device: torch.device = torch.device("cuda")):
    test_accuracy = []
    train_accuracy = []
    for epoch_num in tqdm(range(1, num_epochs + 1), desc="fit", leave=False):
        train(model, train_loader, criterion, optimizer, scheduler, device)
        metric = evaluate(model, train_loader, device)
        train_accuracy.append(metric)
        metric = evaluate(model, test_loader, device)
        test_accuracy.append(metric)
        print(f"epoch {epoch_num}: train_accuracy={train_accuracy[-1]}, test_accuracy={test_accuracy[-1]}")

    return train_accuracy, test_accuracy


from functools import reduce


def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)


def model_size(model):
    param_size = 0
    for name, p in model.named_parameters():
        if 'weight_orig' in name:
            m = get_module_by_name(model, name[:-12])
            if hasattr(m, 'weight_mask'):
                param_size_ = torch.count_nonzero(m.weight_mask) * p.element_size()
                param_size += param_size_
            else:
                param_size_ = p.nelement() * p.element_size()
                param_size += param_size_
        else:
            param_size_ = p.nelement() * p.element_size()
            param_size += param_size_

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size) / 1024 ** 2
    return 'model size: {:.3f}MB'.format(size_all_mb)


def YAML_parser(path="timeloop-model.ERT_summary.yaml"):
    current_path = os.getcwd()
    path = os.path.join(current_path, path)
    with open(path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for data in data_loaded['ERT_summary']['table_summary'][0]['actions']:
        if data['name'] == 'read':
            if "max_energy" in data.keys():
                energy_read = data['max_energy']
            else : 
                energy_read = data['energy']  
        if data['name'] == 'write':
            if "max_energy" in data.keys():
                energy_write = data['max_energy']
            else : 
                energy_write = data["energy"]
    return energy_read, energy_write

def YAML_generator(component_class="SRAM", depth=64,
                   width=8, path="common/Q2/arch_base.yaml"):
    current_path = os.getcwd()
    path = os.path.join(current_path, path)
    with open(path, 'r') as file:
        txt = file.read()
        txt_new = txt.replace('$depth$', str(depth))
        txt_new = txt_new.replace('$width$', str(width))
        txt_new = txt_new.replace('$c_class$', component_class)
        with open('arch.yaml', 'w') as output:
            output.write(txt_new)


class DinoCifar(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        features = self.backbone(x).pooler_output
        logits = self.classifier(features)
        return logits
    
def input_activation_hook(model, data):
    # add hook to record the min max value of the activation
    input_activation = {}
    output_activation = {}

    def add_range_recoder_hook(model):
        import functools
        def _record_range(self, x, y, module_name):
            x = x[0]
            input_activation[module_name] = x.detach()
            output_activation[module_name] = y.detach()

        all_hooks = []
        for name, m in model.named_modules():
            if isinstance(m, (Quantized_Conv2d, Quantized_Linear)):
                all_hooks.append(m.register_forward_hook(
                    functools.partial(_record_range, module_name=name)))
        return all_hooks

    hooks = add_range_recoder_hook(model)
    model(data)

    # remove hooks
    for h in hooks:
        h.remove()
    return input_activation, output_activation


def model_to_quant(model, calibration_loader, act_N_bits=8, weight_N_bits=8, method='sym',
                   device=torch.device("cuda"), bitwidth_dict: Dict = None):
    quantized_model = copy.deepcopy(model)

    input_activation, output_activation = input_activation_hook(
        quantized_model,
        next(iter(calibration_loader))[0].to(device)
    )

    for name, m in quantized_model.named_modules():
        if isinstance(m, (Quantized_Conv2d, Quantized_Linear)):
            if name != 'conv1':
                if bitwidth_dict is None:
                    m.weight_N_bits = weight_N_bits
                else:
                    m.weight_N_bits = bitwidth_dict[name]
                m.act_N_bits = act_N_bits

                m.method = method

                act = input_activation[name]
                act_signed = bool(act.min() < 0)

                m.act_signed = act_signed

                if act_signed:
                    act_scale, _ = reset_scale_and_zero_point(act, m.act_N_bits, method='sym')
                else:
                    act_scale, _ = reset_scale_unsigned(act, m.act_N_bits)

                m.input_scale.data = act_scale.to(m.input_scale.device)

    return quantized_model

def integer_linear(input, weight):
    assert input.dtype == torch.int32
    assert weight.dtype == torch.int32

    if 'cpu' in input.device.type:
        output = F.linear(input, weight)
    else:
        output = F.linear(input.float(), weight.float())
        output = output.round().to(torch.int32)
    return output
def integer_conv2d(input, weight, stride, padding, dilation, groups):
    assert input.dtype == torch.int32
    assert weight.dtype == torch.int32

    if 'cpu' in input.device.type:
        output = F.conv2d(input, weight, None, stride, padding, dilation, groups)
    else:
        output = F.conv2d(input.float(), weight.float(), None, stride, padding, dilation, groups)
        output = output.round().to(torch.int32)
    return output
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def linear_quantize(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor,
                    N_bits: int, signed: bool = True) -> torch.Tensor:
    """
    linear uniform quantization for real tensor
    Args:
        input: torch.tensor
        scale: scale factor
        zero_point: zero point
        N_bits: bitwidth
        signed: flag to indicate signed ot unsigned quantization

    Returns:
        quantized_tensor: quantized tensor whose values are integers
    """

    mini, maxi = (-2**(N_bits-1), 2**(N_bits-1) - 1) if signed else (0, 2**N_bits -1)
    quantized_tensor = torch.clip(torch.round(input/scale) - zero_point ,min=mini, max=maxi)
    return quantized_tensor


def linear_dequantize(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """
    linear uniform de-quantization for quantized tensor
    Args:
        input: input quantized tensor
        scale: scale factor
        zero_point: zero point

    Returns:
        reconstructed_tensor: de-quantized tensor whose values are real
    """
    quantized_tensor = scale*(input + zero_point)
    return quantized_tensor



def get_scale(input, N_bits=8):
    """
    extract optimal scale based on statistics of the input tensor.
    Args:
        input: input real tensor
        N_bits: bitwidth
    Returns:
        scale optimal scale
    """
    assert N_bits in [2, 4, 8]
    z_typical = {'2bit': [0.311, 0.678], '4bit': [0.077, 1.013], '8bit': [0.032, 1.085]}
    z = z_typical[f'{N_bits}bit']
    c1, c2 = 1 / z[0], z[1] / z[0]
    Ex_2 = torch.mean(input**2)
    Ex = torch.mean(input.abs())
    alpha = c1*(Ex_2.sqrt()) - c2*Ex
    q_scale = alpha/(2**(N_bits-1))
    return q_scale


def reset_scale_and_zero_point(input: torch.tensor, N_bits: int = 4, method: str = "sym"):
    """
    Args:
        input: input real tensor
        N_bits: bitwidth
        method: choose between sym, asym, SAWB, and heuristic
    Returns:
        scale factor , zero point
    """
    with torch.no_grad():
        if method == 'heuristic':
            # step_size = argmin_{step_size} (MSE[x, x_hat])
            zero_point = torch.tensor(0)
            def MSE(scale):
                scale = torch.tensor(scale)
                input_int = linear_quantize(input.to('cpu'), scale.to('cpu'), zero_point.to('cpu'), N_bits)
                input_q = linear_dequantize(input_int.to('cpu'), scale.to('cpu'), zero_point.to('cpu'))
                return torch.sum((input.to('cpu') - input_q.to('cpu'))**2)

            initial_guess = input.to('cpu').abs().max()/(2**(N_bits-1))
            result = minimize(MSE, initial_guess)
            step_size = torch.tensor(np.squeeze(result.x), dtype=torch.float32)

        elif method == 'SAWB':
            zero_point = torch.tensor(0.)
            step_size = get_scale(input, N_bits)
        elif method == 'sym':
            zero_point = torch.tensor(0.)
            step_size = torch.max(input.abs())/(2**(N_bits-1))
        elif method == 'asym':
            input_shifted = input - torch.min(input)
            step_size = (torch.max(input) - torch.min(input))/(2**(N_bits)-1)
            zero_point = torch.round(torch.min(input)/step_size)

        else:
            raise "didn't find quantization method."

    return step_size, zero_point
  

class _quantize_func_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, N_bits, signed=True):
        """
        Args:
            ctx: a context object that can be used to stash information for backward computation
            input: torch.tensor
            scale: scale factor
            zero_point: zero point
            N_bits: bitwidth
            signed: flag to indicate signed ot unsigned quantization
        Returns:
            quantized_tensor: quantized tensor whose values are integers
        """
        ctx.scale = scale
        quantized_tensor = linear_quantize(input, scale, zero_point, N_bits, signed)
        return quantized_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output/ctx.scale
        return grad_input, None, None, None, None

linear_quantize_STE = _quantize_func_STE.apply


def quantized_linear_function(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                              input_scale: torch.float, weight_scale: torch.float):
    """
    integer only fully connected layer. 
    Note that you are only allowed to use <integer_linear> function!
    Args:
        input: quantized input
        weight: quantized weight
        bias: quantized bias
        input_scale: input scaling factor
        weight_scale: weight scaling factor

    Returns:
        output: output feature
    """

    bias = 0 if bias is None else bias
    output = input_scale*weight_scale*(integer_linear(input, weight) + bias)
    return output


def quantized_conv2d_function(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                              input_scale: torch.float, weight_scale: torch.float, stride,
                              padding, dilation, groups):
    """
    integer only fully connected layer
    Note that you are only allowed to use <integer_conv2d> function!
    Args:
        groups: number of groups
        stride: stride
        dilation: dilation
        padding: padding
        input: quantized input
        weight: quantized weight
        bias: quantized bias
        input_scale: input scaling factor
        weight_scale: weight scaling factor

    Returns:
        output: output feature
    """

    bias = 0 if bias is None else bias
    output = input_scale*weight_scale*(integer_conv2d(input, weight, stride, padding, dilation, groups) + bias)
    return output

def reset_scale_unsigned(input: torch.tensor, N_bits: int = 4):
    with torch.no_grad():
        zero_point = torch.tensor(0.)
        step_size = torch.max(torch.abs(input)) / ((2**(N_bits))-1)
    return step_size, zero_point

class Quantized_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Quantized_Linear, self).__init__(in_features, out_features, bias=bias)

        self.method = 'normal'  # normal, sym, asym, SAWB,
        self.act_N_bits = None
        self.weight_N_bits = None
        self.input_scale = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.decay = .99

    def forward(self, input):
        if self.method == 'normal':
            # default floating point mode.
            return F.linear(input, self.weight, self.bias)
        else:
            # update scale and zero
            self.__reset_scale_and_zero__(input)
            zero_point = torch.tensor(0.)
            # compute quantized
            signed_act = getattr(self, "act_signed", False)

            quantized_weight = linear_quantize_STE(self.weight, self.weight_scale, zero_point, self.weight_N_bits,True)
            quantized_input = linear_quantize_STE(input, self.input_scale, zero_point, self.act_N_bits, signed_act)
            if self.bias is None:
                quantized_bias = None
            else:
                quantized_bias = linear_quantize_STE(self.bias, self.weight_scale * self.input_scale, zero_point, 32).to(torch.int32)
            output = quantized_linear_function(quantized_input.to(torch.int32), quantized_weight.to(torch.int32),
                                               quantized_bias, self.input_scale, self.weight_scale)
            input_reconstructed = linear_dequantize(quantized_input, self.input_scale, zero_point)
            weight_reconstructed = linear_dequantize(quantized_weight, self.weight_scale, zero_point)
            simulated_output = F.linear(input_reconstructed, weight_reconstructed, self.bias)
            return output + simulated_output - simulated_output.detach()

    def __reset_scale_and_zero__(self, input):
        """
        update scale factor and zero point
            Args:
                input: input feature
            Returns:
        """
        if self.training:
            input_scale_update, _ = reset_scale_unsigned(input, self.act_N_bits)
            self.input_scale.data -= (1 - self.decay) * (self.input_scale - input_scale_update)
        self.weight_scale.data, _= reset_scale_and_zero_point(self.weight, self.weight_N_bits, self.method)


class Quantized_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Quantized_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                               dilation=dilation, groups=groups, bias=bias)
        self.method = 'normal'  # normal, sym, asym, SAWB,
        self.act_N_bits = None
        self.weight_N_bits = None
        self.input_scale = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.decay = .99

    def forward(self, input):
        if self.method == 'normal':
            # default floating point mode.
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            # update scale and zero
            self.__reset_scale_and_zero__(input)
            zero_point = torch.tensor(0.)
            # compute quantized
            signed_act = getattr(self, "act_signed", False)

            quantized_weight = linear_quantize_STE(self.weight, self.weight_scale, zero_point, self.weight_N_bits,True)
            quantized_input = linear_quantize_STE(input, self.input_scale, zero_point, self.act_N_bits, signed_act)
            if self.bias is None:
                quantized_bias = None
            else:
                quantized_bias = linear_quantize_STE(self.bias, self.weight_scale * self.input_scale, zero_point, 32).to(torch.int32)
            output = quantized_conv2d_function(quantized_input.to(torch.int32), quantized_weight.to(torch.int32),
                                               quantized_bias, self.input_scale, self.weight_scale, self.stride,
                                               self.padding, self.dilation, self.groups)
            input_reconstructed = linear_dequantize(quantized_input, self.input_scale, zero_point)
            weight_reconstructed = linear_dequantize(quantized_weight, self.weight_scale, zero_point)
            simulated_output = F.conv2d(input_reconstructed, weight_reconstructed, self.bias, self.stride, self.padding,
                                        self.dilation, self.groups)
            return output + simulated_output - simulated_output.detach()

    def __reset_scale_and_zero__(self, input):
        """
        update scale factor and zero point
            Args:
                input: input feature
            Returns:
        """
        if self.training:
            input_scale_update, _ = reset_scale_unsigned(input, self.act_N_bits)
            self.input_scale.data -= (1 - self.decay) * (self.input_scale - input_scale_update)
        self.weight_scale.data, _ = reset_scale_and_zero_point(self.weight, self.weight_N_bits, self.method)
