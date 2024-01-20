#!/usr/bin/env python3
# Time-stamp: "2024-01-20 13:44:30 (ywatanabe)"

from functools import partial

import mngs
import torch
import torch.nn as nn


class FeatureExtractorTorch(nn.Module):
    def __init__(
        self,
        samp_rate,
        features_list=[
            "mean",
            "std",
            "skewness",
            "kurtosis",
            "median",
            "q25",
            "q75",
            "rms",
            "rfft_bands",
            "beyond_r_sigma_ratio",
        ],
        batch_size=None,
    ):
        super().__init__()

        self.func_dict = dict(
            mean=mngs.dsp.mean_np_torch,
            std=mngs.dsp.std_np_torch,
            skewness=mngs.dsp.skewness_torch,
            kurtosis=mngs.dsp.kurtosis_torch,
            median=mngs.dsp.median_torch,
            q25=mngs.dsp.q25_torch,
            q75=mngs.dsp.q75_torch,
            rms=mngs.dsp.rms_torch,
            rfft_bands=partial(mngs.dsp.rfft_bands_torch, samp_rate=samp_rate),
            beyond_r_sigma_ratio=mngs.dsp.beyond_r_sigma_ratio_torch,
        )

        self.features_list = features_list

        self.batch_size = batch_size

    def forward(self, x):
        if self.batch_size is None:
            conc = torch.cat(
                [self.func_dict[f_str](x) for f_str in self.features_list],
                dim=-1,
            )
            return conc
        else:
            conc = []
            n_batches = len(x) // self.batch_size + 1
            for i_batch in range(n_batches):
                try:
                    start = i_batch * self.batch_size
                    end = (i_batch + 1) * self.batch_size
                    conc.append(
                        torch.cat(
                            [
                                self.func_dict[f_str](x[start:end].cuda())
                                for f_str in self.features_list
                            ],
                            dim=-1,
                        ).cpu()
                    )
                except Exception as e:
                    print(e)
            return torch.cat(conc)


def main():
    BS = 32
    N_CHS = 19
    SEQ_LEN = 2000
    SAMP_RATE = 1000
    # NYQ = int(SAMP_RATE / 2)

    x = torch.randn(BS, N_CHS, SEQ_LEN).cuda()

    m = FeatureExtractorTorch(SAMP_RATE, batch_size=8)

    out = m(x)  # 15 features


if __name__ == "__main__":
    main()
