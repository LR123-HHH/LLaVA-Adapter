#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ============================================================
# DHCCA
# ============================================================

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
    LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch_new_v01 import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size

        # Generative Head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Classification Head
        self.num_classes = getattr(config, 'num_classes', getattr(config, 'num_labels', 3))
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)

        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        sentiment_labels: Optional[torch.LongTensor] = None,
        lambda_cls: float = 0.5,
        #  SAPP 
        sam_output: Optional[list] = None,
        sapp_threshold: int = 5,
        image_features: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                sam_output=sam_output,
                sapp_threshold=sapp_threshold,
                image_features=image_features,
            )

        if sentiment_labels is not None:
            output_hidden_states = True

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,   # True
        )

        # DHCCA
        if sentiment_labels is not None:
            
            gen_loss = outputs.loss   # LLM cross-entropy loss


            last_hidden_states = outputs.hidden_states[-1]   # [B, seq_len, d]
            sample_repr = last_hidden_states[:, -1, :]       # [B, d]
            cls_logits = self.classifier(sample_repr.to(self.classifier.weight.dtype))  # [B, num_classes]
            cls_loss = F.cross_entropy(cls_logits, sentiment_labels)

            # Joint loss
            if gen_loss is not None:
                total_loss = gen_loss + lambda_cls * cls_loss
            else:
                total_loss = lambda_cls * cls_loss

            return CausalLMOutputWithPast(
                loss=total_loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        return outputs

    @torch.no_grad()
    def predict_sentiment(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        sam_output: Optional[list] = None,
        sapp_threshold: int = 5,
        tau: float = 0.97,
        label_map: Optional[dict] = None,
        tokenizer=None,
        max_new_tokens: int = 20,
        **generate_kwargs,
    ) -> List[int]:

        #DHCCA
        if label_map is None:
            label_map = {'positive': 2, 'neutral': 1, 'negative': 0}

        if inputs_embeds is None and images is not None:
            (
                input_ids,
                _,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, None, attention_mask, None, None,
                images, image_sizes,
                sam_output=sam_output, sapp_threshold=sapp_threshold,
            )

        # Prediction by the classification head
    
        lm_outputs = super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = lm_outputs.hidden_states[-1][:, -1, :]   # [B, d]
        # cast to match classifier weight dtype (newly init'd params may differ from fp16 backbone)
        cls_logits = self.classifier(last_hidden.to(self.classifier.weight.dtype))
        cls_probs = F.softmax(cls_logits, dim=-1)               # [B, num_classes]
        cls_conf, cls_pred = cls_probs.max(dim=-1)              # [B], [B]

        # Prediction by the generative head

        gen_ids = LlamaForCausalLM.generate(
            self,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        # Parse the output of the generative head to extract the sentiment polarity
        gen_preds = []
        for seq in gen_ids:
            if tokenizer is not None:
                gen_text = tokenizer.decode(seq, skip_special_tokens=True).lower().strip()
            else:
                gen_text = ''


            matched_label = None
            for keyword, label_id in label_map.items():
                if keyword in gen_text:
                    matched_label = label_id
                    break
            gen_preds.append(matched_label)

        # 3
        predictions = []
        for b in range(cls_conf.shape[0]):
            conf = cls_conf[b].item()
            cls_result = cls_pred[b].item()
            gen_result = gen_preds[b]   # None

            if conf >= tau:
                predictions.append(cls_result)
            else:
                if gen_result is None:
                    predictions.append(cls_result)
                elif gen_result == cls_result:
                    predictions.append(cls_result)
                else:
                    predictions.append(gen_result)

        return predictions

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        sam_output: Optional[list] = None,
        sapp_threshold: int = 5,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                sam_output=sam_output,
                sapp_threshold=sapp_threshold,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, **kwargs,
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
