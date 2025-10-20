import os
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from models.sigmos.compare_sigmos_stft import SignalProcessor

# Basispfad relativ zu dieser Datei
BASE_DIR = os.path.dirname(__file__)
WEIGHT_DIR = os.path.join(BASE_DIR, "sigmos_weights")

model_path = os.path.join(BASE_DIR, "model-sigmos_1697718653_41d092e8-epo-200.onnx")


def load_weight(filename, transpose=False):
    """Lädt ein einzelnes Weight-Array aus dem sigmos_weights-Ordner"""
    full_path = os.path.join(WEIGHT_DIR, os.path.basename(filename))
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Weight file not found: {full_path}")
    weight = np.load(full_path)
    if transpose:
        weight = np.transpose(weight, (1, 0))
    return torch.nn.Parameter(torch.tensor(weight, dtype=torch.float32))


def load_gru_weight():
    """Lädt die GRU-Gewichte (W, R, B)"""
    W_path = os.path.join(WEIGHT_DIR, "onnx__GRU_268.npy")
    R_path = os.path.join(WEIGHT_DIR, "onnx__GRU_269.npy")
    B_path = os.path.join(WEIGHT_DIR, "onnx__GRU_267.npy")

    if not all(os.path.exists(p) for p in [W_path, R_path, B_path]):
        raise FileNotFoundError("Eine oder mehrere GRU-Gewichtsdateien fehlen im sigmos_weights-Ordner.")

    W = torch.tensor(np.load(W_path))
    R = torch.tensor(np.load(R_path))
    B = torch.tensor(np.load(B_path))
    return W, R, B


class SigMOSEstimator(torch.nn.Module):
    def __init__(self):
        super(SigMOSEstimator, self).__init__()
        # input: float32[batch_size,3,time,481]
        # output: float32[batch_size,Addoutput_dim_1]
        self.conv_0 = torch.nn.Conv2d(
            3, 16, [3, 3], padding=1, stride=[1, 1], dilation=[1, 1]
        )
        self.conv_0.weight = load_weight("sigmos_weights/encoder_0_0_weight.npy")
        self.conv_0.bias = load_weight("sigmos_weights/encoder_0_0_bias.npy")
        self.leaky_relu_0 = torch.nn.LeakyReLU(0.009999999776482582)
        self.conv_1 = torch.nn.Conv2d(
            16, 16, [3, 3], padding=1, stride=[1, 1], dilation=[1, 1]
        )
        self.conv_1.weight = load_weight(
            "sigmos_weights/encoder_0_3_block_0_weight.npy"
        )
        self.conv_1.bias = load_weight("sigmos_weights/encoder_0_3_block_0_bias.npy")
        self.elu_0 = torch.nn.ELU(alpha=1.0)
        self.max_pool_0 = torch.nn.MaxPool2d([2, 2], stride=[2, 2])
        self.conv_2 = torch.nn.Conv2d(
            16, 32, [3, 3], padding=1, stride=[1, 1], dilation=[1, 1]
        )
        self.conv_2.weight = load_weight("sigmos_weights/encoder_1_0_weight.npy")
        self.conv_2.bias = load_weight("sigmos_weights/encoder_1_0_bias.npy")
        self.leaky_relu_1 = torch.nn.LeakyReLU(0.009999999776482582)
        self.conv_3 = torch.nn.Conv2d(
            32, 32, [3, 3], padding=1, stride=[1, 1], dilation=[1, 1]
        )
        self.conv_3.weight = load_weight(
            "sigmos_weights/encoder_1_3_block_0_weight.npy"
        )
        self.conv_3.bias = load_weight("sigmos_weights/encoder_1_3_block_0_bias.npy")
        self.elu_1 = torch.nn.ELU(alpha=1.0)
        self.max_pool_1 = torch.nn.MaxPool2d([2, 2], stride=[2, 2])
        self.conv_4 = torch.nn.Conv2d(
            32, 64, [3, 3], padding=1, stride=[1, 1], dilation=[1, 1]
        )
        self.conv_4.weight = load_weight("sigmos_weights/encoder_2_0_weight.npy")
        self.conv_4.bias = load_weight("sigmos_weights/encoder_2_0_bias.npy")
        self.leaky_relu_2 = torch.nn.LeakyReLU(0.009999999776482582)
        self.conv_5 = torch.nn.Conv2d(
            64, 64, [3, 3], padding=1, stride=[1, 1], dilation=[1, 1]
        )
        self.conv_5.weight = load_weight(
            "sigmos_weights/encoder_2_3_block_0_weight.npy"
        )
        self.conv_5.bias = load_weight("sigmos_weights/encoder_2_3_block_0_bias.npy")
        self.elu_2 = torch.nn.ELU(alpha=1.0)
        self.max_pool_2 = torch.nn.MaxPool2d([2, 2], stride=[2, 2])
        self.conv_6 = torch.nn.Conv2d(
            64, 96, [3, 3], padding=1, stride=[1, 1], dilation=[1, 1]
        )
        self.conv_6.weight = load_weight("sigmos_weights/encoder_3_0_weight.npy")
        self.conv_6.bias = load_weight("sigmos_weights/encoder_3_0_bias.npy")
        self.leaky_relu_3 = torch.nn.LeakyReLU(0.009999999776482582)
        self.conv_7 = torch.nn.Conv2d(
            96, 96, [3, 3], padding=1, stride=[1, 1], dilation=[1, 1]
        )
        self.conv_7.weight = load_weight(
            "sigmos_weights/encoder_3_3_block_0_weight.npy"
        )
        self.conv_7.bias = load_weight("sigmos_weights/encoder_3_3_block_0_bias.npy")
        self.elu_3 = torch.nn.ELU(alpha=1.0)
        self.max_pool_3 = torch.nn.MaxPool2d([2, 2], stride=[2, 2])
        self.hidden_size = 320
        self.gru_0 = torch.nn.GRU(
            2880, self.hidden_size, 1, batch_first=False, bidirectional=True
        )

        # W: [2,960,2880] R: [2,960,320] B: [2,1920]
        W, R, B = load_gru_weight()
        # Assign the weights
        # self.gru_0.weight_ih_l0.data = W[0, :, :]
        # self.gru_0.weight_hh_l0.data = R[0, :, :]
        # self.gru_0.bias_ih_l0.data = B[0, :960]
        # self.gru_0.bias_hh_l0.data = B[0, 960:]

        # self.gru_0.weight_ih_l0_reverse.data = W[1, :, :]
        # self.gru_0.weight_hh_l0_reverse.data = R[1, :, :]
        # self.gru_0.bias_ih_l0_reverse.data = B[1, :960]
        # self.gru_0.bias_hh_l0_reverse.data = B[1, 960:]

        # self.gru_0.weight_ih_l0.data = torch.cat(
        #     [W[0, :, :960], W[0, :, 960:1920], W[0, :, 1920:2880]], dim=0
        # )
        # self.gru_0.weight_hh_l0.data = R[0, :, :]
        # self.gru_0.bias_ih_l0.data = B[0, :960]
        # self.gru_0.bias_hh_l0.data = B[0, 960:]

        # self.gru_0.weight_ih_l0_reverse.data = torch.cat(
        #     [W[1, :, :960], W[1, :, 960:1920], W[1, :, 1920:2880]], dim=0
        # )
        # self.gru_0.weight_hh_l0_reverse.data = R[1, :, :]
        # self.gru_0.bias_ih_l0_reverse.data = B[1, :960]
        # self.gru_0.bias_hh_l0_reverse.data = B[1, 960:]

        self.gru_0.weight_ih_l0.data[: self.hidden_size, :] = W[
            0, self.hidden_size : 2 * self.hidden_size, :
        ]  # W_z
        self.gru_0.weight_ih_l0.data[self.hidden_size : 2 * self.hidden_size, :] = W[
            0, : self.hidden_size, :
        ]  # W_r
        self.gru_0.weight_ih_l0.data[2 * self.hidden_size :, :] = W[
            0, 2 * self.hidden_size :, :
        ]  # W_n

        self.gru_0.weight_hh_l0.data[: self.hidden_size, :] = R[
            0, self.hidden_size : 2 * self.hidden_size, :
        ]  # R_z
        self.gru_0.weight_hh_l0.data[self.hidden_size : 2 * self.hidden_size, :] = R[
            0, : self.hidden_size, :
        ]  # R_r
        self.gru_0.weight_hh_l0.data[2 * self.hidden_size :, :] = R[
            0, 2 * self.hidden_size :, :
        ]  # R_n

        self.gru_0.bias_ih_l0.data[: self.hidden_size] = B[
            0, self.hidden_size : 2 * self.hidden_size
        ]  # Wb_z
        self.gru_0.bias_ih_l0.data[self.hidden_size : 2 * self.hidden_size] = B[
            0, : self.hidden_size
        ]  # Wb_r
        self.gru_0.bias_ih_l0.data[2 * self.hidden_size :] = B[
            0, 2 * self.hidden_size : 3 * self.hidden_size
        ]  # Wb_n

        self.gru_0.bias_hh_l0.data[: self.hidden_size] = B[
            0, 4 * self.hidden_size : 5 * self.hidden_size
        ]  # Rb_z
        self.gru_0.bias_hh_l0.data[self.hidden_size : 2 * self.hidden_size] = B[
            0, 3 * self.hidden_size : 4 * self.hidden_size
        ]  # Rb_r
        self.gru_0.bias_hh_l0.data[2 * self.hidden_size :] = B[
            0, 5 * self.hidden_size :
        ]  # Rb_n

        # For the reverse direction (direction 1)
        self.gru_0.weight_ih_l0_reverse.data[: self.hidden_size, :] = W[
            1, self.hidden_size : 2 * self.hidden_size, :
        ]  # W_z
        self.gru_0.weight_ih_l0_reverse.data[
            self.hidden_size : 2 * self.hidden_size, :
        ] = W[
            1, : self.hidden_size, :
        ]  # W_r
        self.gru_0.weight_ih_l0_reverse.data[2 * self.hidden_size :, :] = W[
            1, 2 * self.hidden_size :, :
        ]  # W_n

        self.gru_0.weight_hh_l0_reverse.data[: self.hidden_size, :] = R[
            1, self.hidden_size : 2 * self.hidden_size, :
        ]  # R_z
        self.gru_0.weight_hh_l0_reverse.data[
            self.hidden_size : 2 * self.hidden_size, :
        ] = R[
            1, : self.hidden_size, :
        ]  # R_r
        self.gru_0.weight_hh_l0_reverse.data[2 * self.hidden_size :, :] = R[
            1, 2 * self.hidden_size :, :
        ]  # R_n

        self.gru_0.bias_ih_l0_reverse.data[: self.hidden_size] = B[
            1, self.hidden_size : 2 * self.hidden_size
        ]  # Wb_z
        self.gru_0.bias_ih_l0_reverse.data[self.hidden_size : 2 * self.hidden_size] = B[
            1, : self.hidden_size
        ]  # Wb_r
        self.gru_0.bias_ih_l0_reverse.data[2 * self.hidden_size :] = B[
            1, 2 * self.hidden_size : 3 * self.hidden_size
        ]  # Wb_n

        self.gru_0.bias_hh_l0_reverse.data[: self.hidden_size] = B[
            1, 4 * self.hidden_size : 5 * self.hidden_size
        ]  # Rb_z
        self.gru_0.bias_hh_l0_reverse.data[self.hidden_size : 2 * self.hidden_size] = B[
            1, 3 * self.hidden_size : 4 * self.hidden_size
        ]  # Rb_r
        self.gru_0.bias_hh_l0_reverse.data[2 * self.hidden_size :] = B[
            1, 5 * self.hidden_size :
        ]  # Rb_n

        self.global_max_pooling_0 = torch.nn.AdaptiveMaxPool1d(1)

        self.fc_0 = torch.nn.Linear(1280, 160)
        self.fc_0.weight = load_weight("sigmos_weights/fc1_0_weight.npy")
        self.fc_0.bias = load_weight("sigmos_weights/fc1_0_bias.npy")

        self.leaky_relu_0 = torch.nn.LeakyReLU(0.009999999776482582)

        self.fc_1 = torch.nn.Linear(160, 7)
        self.fc_1.weight = load_weight("sigmos_weights/fc2_0_weight.npy")
        self.fc_1.bias = load_weight("sigmos_weights/fc2_0_bias.npy")

        self.sigmoid_0 = torch.nn.Sigmoid()

        # STFT params
        self.dft_size = 960
        self.frame_size = 480
        self.window_length = 960
        self.compress_factor = 0.3
        window_tensor = torch.sqrt(torch.hann_window(self.window_length))
        self.register_buffer("window", window_tensor)
        # last_frame = len(signal) % self.frame_size
        # if last_frame == 0:
        #     last_frame = self.frame_size

    def compressed_mag_complex(self, x: torch.Tensor):
        x = torch.view_as_real(x)  # -> (B, F, T, 2)
        x = x.permute(0, 2, 3, 1)  # -> (B, T, 2, F)
        x2 = torch.maximum(
            (x * x).sum(dim=-2, keepdim=True),
            torch.tensor(1e-12, dtype=x.dtype, device=x.device),
        )

        if self.compress_factor == 1:
            mag = torch.sqrt(x2)
        else:
            x = torch.pow(x2, (self.compress_factor - 1) / 2) * x
            mag = torch.pow(x2, self.compress_factor / 2)

        features = torch.cat((mag, x), dim=-2)
        features = features.permute(0, 2, 1, 3)
        return features

    def forward(self, input):
        # x = torch.stft(
        #     input,
        #     n_fft=self.dft_size,
        #     hop_length=self.frame_size,
        #     win_length=self.window_length,
        #     window=self.window,
        #     center=True,
        #     pad_mode="constant",
        #     normalized=False,
        #     return_complex=True,
        # )
        # x = self.compressed_mag_complex(x)
        x = input
        x = self.conv_0(x)
        x = self.leaky_relu_0(x)
        b1 = self.conv_1(x)
        b1 = self.elu_0(b1)
        x = x + b1
        x = self.max_pool_0(x)
        x = self.conv_2(x)
        x = self.leaky_relu_1(x)
        b1 = self.conv_3(x)
        b1 = self.elu_1(b1)
        x = x + b1
        x = self.max_pool_1(x)
        x = self.conv_4(x)
        x = self.leaky_relu_2(x)
        b1 = self.conv_5(x)
        b1 = self.elu_2(b1)
        x = x + b1
        x = self.max_pool_2(x)
        x = self.conv_6(x)
        x = self.leaky_relu_3(x)
        b1 = self.conv_7(x)
        b1 = self.elu_3(b1)
        x = x + b1
        x = self.max_pool_3(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x = x.permute(1, 0, 2)
        x, _ = self.gru_0(x)
        x_max = x.permute(1, 2, 0)
        x_max = self.global_max_pooling_0(x_max).squeeze(-1)
        # x = x[:, -1, :]
        x_mean = x.permute(1, 0, 2)
        x_mean = torch.mean(x_mean, dim=1)
        x = torch.cat((x_max, x_mean), dim=1)
        x = self.fc_0(x)
        x = self.leaky_relu_0(x)
        x = self.fc_1(x)
        x = self.sigmoid_0(x)
        x = x * 4.0
        x = x + 1.0
        return x


def load_model_and_weights():
    model = SigMOSEstimator()
    torch.onnx.export(
        model, torch.randn(1, 3, 500, 481), "sigmos_build.onnx", verbose=False
    )


def compare_models():
    model_path = "model-sigmos_1697718653_41d092e8-epo-200_transpose24.onnx"
    stft = SignalProcessor()
    input = torch.randn(1, 48000 * 5)
    stft_list = []
    for i in range(input.size(0)):
        stft_result = stft.stft(np.array(input[i].detach().numpy()).astype("float32"))
        compressed_result = stft.compressed_mag_complex(stft_result)
        stft_list.append(compressed_result)
    compressed_result = np.concatenate(stft_list, axis=0)
    print(compressed_result.shape)

    onnx_sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    model_input = {"input": compressed_result}
    onnx_result = onnx_sess.run(["output"], model_input)[0]
    print("ONNX Result")
    print(onnx_result.shape)

    model = SigMOSEstimator()
    torch_stft = torch.from_numpy(compressed_result)
    torch.onnx.export(
        model,
        torch_stft,
        "sigmos_model.onnx",
        # verbose=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "time"},
        },
    )
    import onnx

    onnx.shape_inference.infer_shapes_path("sigmos_model.onnx")
    # export torch model
    # torch.save(model.state_dict(), "model-sigmos_1697718653_41d092e8-epo-200.pt")
    results = model(torch_stft)

    # onnx inference
    model_path = "sigmos_model.onnx"
    onnx_sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    model_input = {"input": compressed_result}
    onnx_result = onnx_sess.run(["output"], model_input)[0]

    # print("Torch Result")
    print(results.shape)

    # all close
    print("All close")
    print(torch.allclose(torch.tensor(onnx_result), results, atol=1e-4, rtol=1e-4))


if __name__ == "__main__":
    # model = DNSMOS()
    # model.load_state_dict(torch.load("dnsmosp835_sig_bak_ovr.pth"))
    # model.eval()
    # print(model)
    # print(model(torch.randn(1, 144160)).shape)
    # torch.Size([1, 161, 900])

    # export to onnx
    # torch.onnx.export(model, torch.randn(1, 144160), "dnsmos_build.onnx", verbose=False)

    # compare_models()
    compare_models()
    # dnsmos_model = SigMOSEstimator()
    # dnsmos_model.load_state_dict(torch.load("dnsmosp835_sig_bak_ovr.pt"))
    # # del dnsmos_model._modules["fc3"]
    # print(dnsmos_model(torch.randn(1, 144160)).shape)
