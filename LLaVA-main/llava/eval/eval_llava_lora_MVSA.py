import argparse
import json
import torch
import os
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model import *  # Import the entire module directly to enable function overriding
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
import re, os, sys
from tqdm import tqdm

# Save a reference to the original function
original_load_pretrained_model = llava.model.builder.load_pretrained_model

# Override the original function to handle LoRA scenarios
def load_pretrained_model_for_lora(model_path, model_base, model_name):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    # Load the configuration
    config = AutoConfig.from_pretrained(model_path)
    
    # Load the model without loading the multimodal projector
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16
    )
    
    # Load the image processor
    from transformers import CLIPImageProcessor
    image_processor = CLIPImageProcessor.from_pretrained(
        model.config.mm_vision_tower, torch_dtype=torch.float16
    )
    
    # Compute the context length
    context_len = 2048
    
    return tokenizer, model, image_processor, context_len

# Temporarily replace the function
llava.model.builder.load_pretrained_model = load_pretrained_model_for_lora

# The following is the original evaluation code
def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image_file = os.path.join("/root/autodl-tmp/LLaVA-main/playground/data", image_file)
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(model, model_name, tokenizer, image_processor, args, image_file, query):
    # Model
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs.replace(DEFAULT_IMAGE_TOKEN,"").strip()

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images = load_images([image_file])
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default="/root/autodl-tmp/MVSA_test.json")
    parser.add_argument("--output-file", type=str, default="/root/autodl-tmp/MVSA_test_output.json")
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/llava-v1.5-7b")
    parser.add_argument("--lora-path", type=str, default="/root/autodl-tmp/LLaVA-main/checkpoints/llava-v1.5-7b-sentiment")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    args = parser.parse_args()
    
    disable_torch_init()
    
    # Get the model name
    model_name = get_model_name_from_path(args.model_path)
    
    # Load the base model using the modified function
    tokenizer, model, image_processor, context_len = llava.model.builder.load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    
    # Load LoRA weights if a LoRA path is provided
    if args.lora_path:
        from peft import PeftModel
        print(f"Loading LoRA weights from {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
        print("LoRA weights loaded successfully")
    
    # Read the input file
    with open(args.input_file, 'r') as infile:
        inputs = json.load(infile)

    results = []
    for entry in tqdm(inputs):
        try:
            image_file = entry["image"]
            query = entry["conversations"][0]["value"]
            result = eval_model(model, model_name, tokenizer, image_processor, args, image_file, query)
            entry["model_response"] = result
            results.append(entry)
            print(entry)
            print("=="*10)
        except Exception as e:
            print(f"Error processing sample: {e}")
            entry["model_response"] = "Error processing"
            results.append(entry)
            continue

    # Save the results
    with open(args.output_file, 'w') as outfile:
        json.dump(results, outfile, indent=2)
    
    # Restore the original function
    llava.model.builder.load_pretrained_model = original_load_pretrained_model
    
    print(f"Evaluation completed. Results saved to {args.output_file}")