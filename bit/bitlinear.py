
from torch import nn , Tensor
from torch.nn import functional as F
from .RMSNorm import RMSNorm

def weight_quant(w: Tensor) ->Tensor:
    scale = w.abs().mean()
    e = w.mean()
    u = (w - e).sign() * scale
    return u

def activation_quant(x: Tensor) ->Tensor:
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

class BitLinear(nn.Linear):
    """
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    """
    def __init__(self, in_features, out_features, bias = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.norm = RMSNorm(self.in_features)
    def forward(self,x):
        """
        Args:
            x: input tensor
        """
        w = self.weight
        x_norm = self.norm(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        
        return F.linear(x_quant,w_quant)
        
# def replace_linear(module):
#     if isinstance(module, nn.Linear):
#         # 替换为 BitLinear
#         bitlinear = BitLinear(module.in_features, module.out_features, module.bias is not None)
#         # 复制权重和偏置
#         bitlinear.weight = module.weight
#         if module.bias is not None:
#             bitlinear.bias = module.bias
#         return bitlinear
#     else:
#         return module

def _replace_with_bitnet_linear(model, modules_to_not_convert=None, current_key_name=None, isSuccess=False):
    """
    递归地替换模型中的所有 nn.Linear 层为 BitLinear 层。
    
    参数：
    - model: 需要替换层的 PyTorch 模型。
    - modules_to_not_convert: 需要排除的模块名称列表。
    - current_key_name: 当前模块的名称，用于递归。
    - isSuccess: 是否替换成功
    """
    if current_key_name is None:
        current_key_name = []

    # 遍历模型中的每个子模块
    for name, module in model.named_children():
        # 更新当前的模块名称
        current_key_name.append(name)

        # 检查该模块是否需要替换
        if not any(key in ".".join(current_key_name) for key in (modules_to_not_convert or [])):
            if isinstance(module, nn.Linear):
                # 如果是 Linear 层，替换为 BitLinear 层
                in_features = module.in_features
                out_features = module.out_features
                model._modules[name] = BitLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=module.bias is not None
                )
                isSuccess = True
        
        # 如果当前模块有子模块，则递归替换
        if len(list(module.children())) > 0:
            _replace_with_bitnet_linear(module, modules_to_not_convert, current_key_name)

        # 递归结束后移除当前模块的名称
        current_key_name.pop()

    return model , isSuccess

def replace_with_bitnet_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
):
    """
    A helper function to replace all `torch.nn.Linear` modules by `BitLinear` modules`.
    """
    isSuccess = False
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    modules_to_not_convert = list(set(modules_to_not_convert))
    model , isSuccess= _replace_with_bitnet_linear(
        model,
        modules_to_not_convert,
        current_key_name,
        isSuccess
    )
    if isSuccess:
        print("成功替换了所有 nn.Linear 层为 BitLinear 层。")
    else:
        print("没有找到 nn.Linear 层 或 没有替换成功。")
    return model
