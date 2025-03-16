
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

    def forward(self,x):
        """
        Args:
            x: input tensor
        """
        w = self.weight
        norm = RMSNorm(self.in_features)
        x_norm = norm(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        
        return F.linear(x_quant,w_quant)
        

