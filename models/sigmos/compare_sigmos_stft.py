import librosa
import numpy as np
import scipy
import torch


class SignalProcessor:
    def __init__(self):
        self.sampling_rate = 48_000
        self.resample_type = "fft"

        # STFT params
        self.dft_size = 960
        self.frame_size = 480
        self.window_length = 960
        self.window = np.sqrt(np.hanning(int(self.window_length) + 1)[:-1]).astype(
            np.float32
        )

    def stft(self, signal):
        last_frame = len(signal) % self.frame_size
        if last_frame == 0:
            last_frame = self.frame_size

        padded_signal = np.pad(
            signal,
            ((self.window_length - self.frame_size, self.window_length - last_frame),),
        )
        frames = librosa.util.frame(
            padded_signal,
            frame_length=len(self.window),
            hop_length=self.frame_size,
            axis=0,
        )
        spec = scipy.fft.rfft(frames * self.window, n=self.dft_size)
        return spec.astype(np.complex64)

    def compressed_mag_complex(self, x: np.ndarray, compress_factor=0.3):
        x = x.view(np.float32).reshape(x.shape + (2,)).swapaxes(-1, -2)
        x2 = np.maximum((x * x).sum(axis=-2, keepdims=True), 1e-12)
        if compress_factor == 1:
            mag = np.sqrt(x2)
        else:
            x = np.power(x2, (compress_factor - 1) / 2) * x
            mag = np.power(x2, compress_factor / 2)

        features = np.concatenate((mag, x), axis=-2)
        features = np.transpose(features, (1, 0, 2))
        return np.expand_dims(features, 0)


def compressed_mag_complex_torch(x: torch.Tensor, compress_factor=0.3):
    x = torch.view_as_real(x)  # -> (B, F, T, 2)
    x = x.permute(0, 2, 3, 1)  # -> (B, T, 2, F)
    x2 = torch.maximum(
        (x * x).sum(dim=-2, keepdim=True),
        torch.tensor(1e-12, dtype=x.dtype, device=x.device),
    )

    if compress_factor == 1:
        mag = torch.sqrt(x2)
    else:
        x = torch.pow(x2, (compress_factor - 1) / 2) * x
        mag = torch.pow(x2, compress_factor / 2)

    features = torch.cat((mag, x), dim=-2)
    features = features.permute(0, 2, 1, 3)
    return features


# Testing with random data
if __name__ == "__main__":
    model_version = "1.0"
    processor = SignalProcessor(model_version)

    # Generate random signal data
    random_signal = np.random.randn(48000).astype(
        np.float32
    )  # 1 second of random data at 48kHz

    # Perform STFT
    stft_result = processor.stft(random_signal)
    print("STFT Result:", stft_result.shape)

    # Perform compression
    compressed_result = processor.compressed_mag_complex(stft_result)
    print("Compressed Result:", compressed_result.shape)

    dft_size = 960
    frame_size = 480
    window_length = 960
    window_tensor = torch.sqrt(torch.hann_window(window_length + 1)[:-1])

    x = torch.stft(
        torch.tensor(random_signal).unsqueeze(0),
        n_fft=dft_size,
        hop_length=frame_size,
        win_length=window_length,
        window=window_tensor,
        center=True,
        pad_mode="constant",
        normalized=False,
        return_complex=True,
    )

    # x = x.permute(0, 2, 1)

    print("PyTorch STFT Result:", x.shape)

    x = compressed_mag_complex_torch(x)

    # x = torch.view_as_real(x)  # -> (B, F, T, 2)
    # x = x.permute(0, 3, 2, 1)  # -> (B, 2, F, T)

    print("PyTorch STFT Result:", x.shape)
    x_numpy = x.detach().numpy()

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(compressed_result - x_numpy))
    print(f"Mean Absolute Error (MAE): {mae}")

    # Mean Squared Error (MSE)
    mse = np.mean((compressed_result - x_numpy) ** 2)
    print(f"Mean Squared Error (MSE): {mse}")

    # Maximum Absolute Error
    max_abs_error = np.max(np.abs(compressed_result - x_numpy))
    print(f"Maximum Absolute Error: {max_abs_error}")

    # Relative Error
    relative_error = np.mean(
        np.abs((compressed_result - x_numpy) / np.maximum(np.abs(x_numpy), 1e-12))
    )
    print(f"Relative Error: {relative_error}")

    # Allclose for reference
    allclose = np.allclose(compressed_result, x_numpy, atol=1e-1, rtol=1e-1)
    print(f"Allclose: {allclose}")
