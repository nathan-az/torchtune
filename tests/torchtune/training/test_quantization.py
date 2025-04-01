# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from torchao.float8.float8_linear import Float8Linear

from torchtune.models.llama3 import base_llama_tp_plan, fp8_llama_tp_plan
from torchtune.training.quantization import (
    convert_to_float8_training,
    is_fp8_tensorwise_scaling,
    validate_float8_tp_plan,
)


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 256, bias=False).to(torch.float)
        self.output = torch.nn.Linear(256, 512, bias=False).to(torch.float)

    def example_inputs(self):
        return (torch.randn(1, 512).to(torch.float),)

    def forward(self, x):
        x = self.linear(x)
        x = self.output(x)
        return x


class TestFloat8:
    def test_convert_to_float8_training(self):
        """
        Test that target linear layers are converted to Float8Linear.
        """
        m = M()
        example_inputs = torch.randn(1, 512).to(torch.float)
        m = convert_to_float8_training(m)
        assert isinstance(m.linear, Float8Linear)
        assert not isinstance(m.output, Float8Linear)
        with pytest.raises(Exception):
            m = convert_to_float8_training(m, "unrecognized_recipe_name")

    def test_validate_float8_tp_plan(self):
        """
        Test that only float8 TP plan is only valid for "tensorwise" float8 recipes.
        """
        validate_float8_tp_plan(base_llama_tp_plan())
        validate_float8_tp_plan(base_llama_tp_plan(), "anything")
        validate_float8_tp_plan(fp8_llama_tp_plan())
        validate_float8_tp_plan(fp8_llama_tp_plan(), "tensorwise")
        with pytest.raises(ValueError):
            validate_float8_tp_plan(fp8_llama_tp_plan(), "rowwise")
        with pytest.raises(ValueError):
            validate_float8_tp_plan(fp8_llama_tp_plan(), "rowwise_with_gw_hp")

    def test_is_fp8_tensorwise_scaling(self):
        """
        Test that `is_fp8_tensorwise_scaling` returns True only for tensorwise scaling.
        """
        assert is_fp8_tensorwise_scaling(None)
        assert is_fp8_tensorwise_scaling("tensorwise")
        assert not is_fp8_tensorwise_scaling("rowwise")
        assert not is_fp8_tensorwise_scaling("rowwise_with_gw_hp")
