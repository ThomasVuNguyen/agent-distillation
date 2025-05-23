import dataclasses
import importlib.resources as pkg_resources
import json
import random
import warnings
from collections import deque
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any, Literal, Optional, Union

import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from accelerate import Accelerator, PartialState
from accelerate.state import AcceleratorState
from huggingface_hub import ModelCard, ModelCardData
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import (
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EvalPrediction,
    GenerationConfig,
    PreTrainedTokenizerBase,
    TrainerState,
    TrainingArguments,
    is_comet_available,
)
from transformers.utils import (
    is_peft_available,
    is_torch_mlu_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)


class DataCollatorForCompletionOnlyLMMultiTurn(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, list[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, list[int], list[str]]`, *optional*, defaults to `None`):
            The template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids
            or as a list of strings when multiple instruction templates need to be detected (useful for multi-turn conversations e.g. ["<system>", "<tool>", "<user>"]).
        mlm (`bool`, *optional*, defaults to `False`):
            Whether to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
        padding_free (`bool`, *optional*, defaults to `False`):
            Whether to use padding-free training. When set to True, padding tokens are removed and positional ids are
            added to the inputs to enable proper attention.
    """

    def __init__(
        self,
        response_template: Union[str, list[int]],
        instruction_template: Optional[Union[str, list[int], list[str]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        padding_free: bool = False,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        self.has_multiple_instruction_templates = False

        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        elif isinstance(instruction_template, list) and isinstance(instruction_template[0], str):
            # The user provides a list of strings, must tokenize each template
            self.instruction_token_ids = []
            for template in self.instruction_template:
                self.instruction_token_ids.append(self.tokenizer.encode(template, add_special_tokens=False))
            self.has_multiple_instruction_templates = True
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template
            # Check if it's a list of lists (multiple templates)
            if (
                isinstance(instruction_template, list)
                and instruction_template
                and isinstance(instruction_template[0], list)
            ):
                self.has_multiple_instruction_templates = True

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value.",
                UserWarning,
            )

        self.ignore_index = ignore_index
        self.padding_free = padding_free

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)

        sequence_lengths = (batch["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)
        content_starts = (
            batch["input_ids"].shape[1] - sequence_lengths
            if self.tokenizer.padding_side == "left"
            else torch.zeros_like(sequence_lengths)
        )
        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, consider increasing the `max_length`.",
                        UserWarning,
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_start_positions = []
                instruction_start_positions = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_start_positions.append(assistant_idx + len(self.response_token_ids))

                if len(response_start_positions) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, consider increasing the `max_length`.",
                        UserWarning,
                    )
                    batch["labels"][i, :] = self.ignore_index
                    continue

                # Find all instruction token positions
                if self.has_multiple_instruction_templates:
                    # Handle multiple instruction templates
                    for instruction_token_ids in self.instruction_token_ids:
                        for instruction_idx in np.where(batch["labels"][i] == instruction_token_ids[0])[0]:
                            if (
                                instruction_token_ids
                                == batch["labels"][i][
                                    instruction_idx : instruction_idx + len(instruction_token_ids)
                                ].tolist()
                            ):
                                instruction_start_positions.append(instruction_idx)
                    instruction_start_positions = sorted(instruction_start_positions)
                else:
                    instruction_token_ids = self.instruction_token_ids
                    for instruction_idx in np.where(batch["labels"][i] == instruction_token_ids[0])[0]:
                        # find the indexes of the start of an instruction.
                        if (
                            instruction_token_ids
                            == batch["labels"][i][
                                instruction_idx : instruction_idx + len(instruction_token_ids)
                            ].tolist()
                        ):
                            instruction_start_positions.append(instruction_idx)

                if len(instruction_start_positions) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, consider increasing the `max_length`.",
                        UserWarning,
                    )
                    batch["labels"][i, :] = self.ignore_index
                    continue

                # Mask everything first and we will unmask step by step
                batch["labels"][i, :] = self.ignore_index

                # Unmask regions between each response and next instruction (or till end)
                sequence_length = sequence_lengths[i].item()
                content_start = content_starts[i].item()
                last_processed_instruction_pos = -1
                for response_pos in response_start_positions:
                    # Find the first instruction position that comes after this response
                    next_instruction_pos = None
                    for instruction_pos in instruction_start_positions:
                        if instruction_pos > response_pos:
                            next_instruction_pos = instruction_pos
                            break

                    # If no instruction position found after response, use sequence length from input_ids
                    if next_instruction_pos is None:
                        # Calculate actual sequence length using pad token positions
                        next_instruction_pos = content_start + sequence_length

                    # Handle consecutive responses
                    if response_pos > last_processed_instruction_pos:
                        # Unmask from response start to instruction start (or end); base case
                        batch["labels"][i, response_pos:next_instruction_pos] = batch["input_ids"][
                            i, response_pos:next_instruction_pos
                        ]
                        last_processed_instruction_pos = next_instruction_pos
                    else:
                        # 2 reponses in a row so we unmask the special tokens for response in the middle
                        batch["labels"][i, response_pos - len(self.response_token_ids) : response_pos] = (
                            self.ignore_index
                        )

        if self.padding_free:
            # remove padding, `attention_mask` and add `position_ids`
            attn_mask = batch.pop("attention_mask")
            batch["input_ids"] = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
            batch["position_ids"] = attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
            batch["labels"] = batch["labels"][attn_mask.bool()].unsqueeze(0)
            batch["labels"][batch["position_ids"] == 0] = self.ignore_index

            # Calculate cumulative sequence lengths for queries and keys to prevent graph breaks during further computations.
            flattened_position_ids = batch["position_ids"].flatten()
            indices_q = torch.arange(
                flattened_position_ids.size(0), device=flattened_position_ids.device, dtype=torch.int32
            )
            batch["cu_seq_lens_q"] = torch.cat(
                (
                    indices_q[flattened_position_ids == 0],
                    torch.tensor(
                        flattened_position_ids.size(), device=flattened_position_ids.device, dtype=torch.int32
                    ),
                )
            ).unsqueeze(0)
            batch["cu_seq_lens_k"] = batch["cu_seq_lens_q"]

            # Determine maximum sequence lengths to prevent graph breaks during further computations.
            batch["max_length_k"] = torch.tensor([flattened_position_ids.max().item() + 1])
            batch["max_length_q"] = batch["max_length_k"]

        # # Let's analyze this
        # labels = batch["labels"][0] # List of label
        # input_ids = batch["input_ids"][0]
        # # Apply mask
        # mask = labels != -100
        # filtered_input_ids = input_ids[mask]

        # replaced_labels = [label if label != -100 else 0 for label in labels]
        # replaced_inversed_labels = [input_ids[i] if label == -100 else 0 for i, label in enumerate(labels)]

        # print("Filtered input_ids:", filtered_input_ids.tolist())
        # print(self.tokenizer.decode(replaced_labels))
        # print()
        # print(self.tokenizer.decode(replaced_inversed_labels))
        # import pdb; pdb.set_trace()

        return batch