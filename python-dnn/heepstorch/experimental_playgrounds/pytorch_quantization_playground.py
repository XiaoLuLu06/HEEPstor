import torch
import torch.nn as nn
import torch.quantization as quant


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 2)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fc.weight.data *= 100
        self.fc.weight.data += 10

    def forward(self, x):
        return self.fc(x)


# Configure model for weight-only quantization
def prepare_model(model):
    # Configure for static quantization

    per_channel_symmetric_obs = quant.PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        quant_min=-127,
        quant_max=127,
        eps=2 ** -12,
    )

    per_channel_asymmetric_obs = quant.PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_affine,
        quant_min=-127,
        quant_max=127,
        eps=2 ** -12,
    )

    per_tensor_symmetric_obs = quant.PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        quant_min=-127,
        quant_max=127,
        eps=2 ** -12,
    )

    per_tensor_asymmetric_obs = quant.PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine,
        quant_min=-127,
        quant_max=127,
        eps=2 ** -12,
    )

    # TODO: Also see per_channel_affine_float_qparams (maybe with floating point zero-point)

    # fake_input = torch.Tensor([1, 2, 3, 4, 5])
    # print(model(fake_input))

    model.qconfig = quant.QConfig(
        activation=quant.MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            is_dynamic=True
        )
        ,
        weight=per_channel_asymmetric_obs
    )
    # model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # model.qconfig = quant.float_qparams_weight_only_qconfig
    model_prepared = torch.quantization.prepare(model)

    # print(model_prepared(fake_input))

    # Quantize the model
    model_quantized = torch.quantization.convert(model_prepared)

    # TODO: Find ways to evaluate the accuracy of the model. Maybe fake_quantize? Maybe steal the weights?
    # print(model_quantized(fake_input))
    # exit()

    # Access quantization parameters
    # fc_weight = model_quantized.fc._packed_params._packed_params[0]
    # scale = fc_weight.q_scale()
    # zero_point = fc_weight.q_zero_point()

    fc_weight = model_quantized.fc.weight()
    print('Dequantized weight:')
    print(fc_weight.dequantize())

    fc_scale = model_quantized.fc.weight().q_per_channel_scales()  # Scale
    fc_zero_point = model_quantized.fc.weight().q_per_channel_zero_points()  # Zero-point
    fc_int8_weights = model_quantized.fc.weight().int_repr()  # int8 weights

    return model_quantized, fc_scale, fc_zero_point, fc_int8_weights


# Example usage
model = SimpleModel()

fp32_weights = model.fc.weight
print(f'FP32 Weights:')
print(fp32_weights)

model_q, scale, zero_point, int8_weights = prepare_model(model)
print(f"Scale: {scale}, Zero Point: {zero_point}")
print(f'INT8 Weights:')
print(int8_weights)
