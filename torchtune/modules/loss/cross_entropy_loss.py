# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    Partial,
    Placement,
    Shard,
)
from torch.distributed.tensor.parallel import ColwiseParallel

from torchtune.modules.loss.loss_types import SFTLoss
from torchtune.utils import get_logger

log = get_logger()


class LinearCrossEntropyLoss(SFTLoss, nn.Module):
    """Memory efficient Cross-entropy loss that incrementally computes loss for chunks of tokens
    by masking ignored tokens, calculating logits and then applying cross-entropy loss. Combines
    the linear projection with the cross-entropy calculation for further memory savings.

    Linear cross entropy masks out ignored tokens before the projection layer to save memory.
    You therefore need to skip the final projection layer in your model and pass it to the loss instead.
    You can setup the loss with the model and compile it as shown below.

    >>> model = Transformer(...)
    >>> loss = LinearCrossEntropyLoss(...)
    >>> loss.set_model_output(model)
    >>> loss.apply_compile_strategy()
    """

    def __init__(
        self,
        num_output_chunks: int = 8,
        ignore_index: int = -100,
        enable_loss_parallel: bool = False,
    ):
        super().__init__(enable_loss_parallel=enable_loss_parallel)
        """
        Args:
            num_output_chunks (int): Number of chunks to split the output tensor into. Default is 8.
            ignore_index (int): Index to ignore in the target tensor. Default is -100.
        """
        self.linear_projection = None
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index

    def apply_compile_strategy(self, *args, **kwargs):
        """Applies compile only to the compute_cross_entropy function.
        If compiling CE + chunking operation together, memory requirement is higher."""
        # log.warning("Skipping compile loss, as it is not supported at this time")
        # TODO fix compile and re-enable
        self.compute_cross_entropy = torch.compile(
            self.cross_entropy_loss_fn, *args, **kwargs
        )
        return self

    def set_model_output(self, model: nn.Module) -> None:
        """Modify model output to match the expected input for the loss function."""
        model.skip_output_layer = True
        self.linear_projection = model.output

    def patch_tp_plan(self, tp_plan) -> bool:
        if self.loss_parallel_enabled:
            if "output" not in tp_plan:
                raise KeyError("`tp_plan` requires `output` key")

            tp_plan["output"] = ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(0),
                use_local_output=True,
            )
        return tp_plan

    @property
    def supports_loss_parallel(self) -> bool:
        return True

    @property
    def loss_parallel_requires_ctx_manager(self) -> bool:
        return False

    @property
    def cross_entropy_loss_fn(self):
        # just returns branchless versions of loss function options
        if self.loss_parallel_enabled:
            return self.compute_cross_entropy_distributed
        else:
            return self.compute_cross_entropy_local

    def compute_cross_entropy_local(
        self,
        hidden_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Computes cross-entropy by masking tokens, calculating logits and then applying cross-entropy loss.

        Args:
            hidden_chunk (torch.Tensor): [batch_size, chunk_size, embed_dim]
            target_chunk (torch.Tensor): [batch_size, chunk_size]
            **kwargs: allows compatibility with distributed loss function when called selectively

        Returns:
            torch.Tensor: Sum of cross-entropy loss for non-ignored tokens in the chunk

        Raises:
            AttributeError: if called before update_model
        """
        mask_chunk = target_chunk != self.ignore_index
        if mask_chunk.sum() == 0:
            # Unmask 1 token to allow loss to sync with all data parallel workers
            mask_chunk[0] = True

        target_chunk = target_chunk[mask_chunk]  # [num_valid]
        hidden_chunk = hidden_chunk[mask_chunk]  # [num_valid, embed_dim]

        # [num_valid, embed_dim] @ [embed_dim, vocab_size]
        if self.linear_projection is None:
            raise AttributeError("forward called before update_model")
        logits = self.linear_projection(hidden_chunk)  # [num_valid, vocab_size]

        return F.cross_entropy(
            logits.float(),
            target_chunk,
            reduction="sum",
            ignore_index=self.ignore_index,
        )

    def compute_cross_entropy_distributed(
        self,
        hidden_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
        *,
        original_mesh: DeviceMesh | None = None,
        original_placements: list[Placement] | None = None,
    ) -> torch.Tensor:
        """Computes cross-entropy by masking tokens, calculating logits and then applying cross-entropy loss.

        Args:
            hidden_chunk (torch.Tensor): [batch_size, chunk_size, embed_dim]
            target_chunk (torch.Tensor): [batch_size, chunk_size]
            original_mesh (DeviceMesh | None): Device mesh of the original tensor if distributed
            original_placements (list[Placement] | None): Placements of the original tensor if distributed

        Returns:
            torch.Tensor: Sum of cross-entropy loss for non-ignored tokens in the chunk

        Raises:
            AttributeError: if called before update_model
        """
        hidden_chunk = hidden_chunk.reshape(-1, hidden_chunk.shape[-1])
        target_chunk = target_chunk.reshape(-1)

        mask_chunk = target_chunk != self.ignore_index

        if mask_chunk.sum() == 0:
            # Unmask 1 token to allow loss to sync with all data parallel workers
            mask_chunk[0] = True

        target_chunk = target_chunk[mask_chunk]  # [num_valid]
        local_hidden_chunk = hidden_chunk.to_local()[mask_chunk]
        hidden_chunk = DTensor.from_local(
            local_hidden_chunk, original_mesh, original_placements
        )  # [num_valid, embed_dim]

        # [num_valid, embed_dim] @ [embed_dim, vocab_size]
        if self.linear_projection is None:
            raise AttributeError("forward called before update_model")
        logits = self.linear_projection(hidden_chunk)  # [num_valid, vocab_size]

        # used only for actual loss function since it will align to pre-masked size of hidden_chunk
        target_chunk_shard = distribute_tensor(
            target_chunk, original_mesh, [Shard(0)] * original_mesh.ndim
        ).to_local()
        target_chunk_shard = target_chunk_shard[target_chunk_shard != self.ignore_index]

        loss = F.cross_entropy(
            logits.float(),
            target_chunk_shard,
            reduction="sum",
            ignore_index=self.ignore_index,
        )
        # is there a cleaner way to do this?
        return DTensor.from_local(
            loss, original_mesh, [Partial()] * original_mesh.ndim
        ).full_tensor()

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            outputs (torch.Tensor): Hidden state of the model, pre projection. Shape ``[bsz, seq_len, emb_dim]``
            targets (torch.Tensor): Labels for the model. Shape ``[bsz, seq_len]``

        Returns:
            torch.Tensor: loss tensor
        """
        # Total number of non-ignored tokens across the entire batch
        mask = targets != self.ignore_index
        total_elements = mask.sum()

        original_mesh = None
        original_placements = None
        if isinstance(outputs, DTensor):
            original_placements = outputs.placements
            original_mesh = outputs.device_mesh

            # resharding on the feature dim stops the sharding dim (currently sequence) from
            # decaying to Replicate during tensor_split
            outputs = outputs.redistribute(
                device_mesh=original_mesh, placements=[Shard(-1)] * original_mesh.ndim
            )

        hidden_chunks = outputs.tensor_split(self.num_output_chunks, dim=1)
        target_chunks = targets.tensor_split(self.num_output_chunks, dim=1)

        # Compute cross-entropy loss for the chunks
        total_loss = 0.0
        for idx in range(len(hidden_chunks)):
            total_loss += self.cross_entropy_loss_fn(
                hidden_chunks[idx],
                target_chunks[idx],
                original_mesh=original_mesh,
                original_placements=original_placements,
            )

        if total_elements == 0:
            # must return after calling compute_cross_entropy to not hang during data parallel training
            return total_loss
        else:
            return total_loss / total_elements
