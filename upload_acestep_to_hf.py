#!/usr/bin/env python3
"""
Self-contained script to convert ACE-Step v1.5 turbo weights from PyTorch to MLX
and upload them to mlx-community on HuggingFace.

Usage:
    pip install huggingface_hub safetensors torch transformers diffusers
    huggingface-cli login
    python upload_acestep_to_hf.py
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def convert_and_upload(
    model_repo: str = "ACE-Step/Ace-Step1.5",
    hf_repo: str = "mlx-community/ACE-Step-v1.5-turbo-mlx",
    turbo_subdir: str = "acestep-v15-turbo",
    text_encoder_subdir: str = "checkpoints/Qwen3-Embedding-0.6B",
    vae_subdir: str = "vae",
):
    import mlx.core as mx
    import safetensors.torch
    import torch
    from diffusers.models import AutoencoderOobleck
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()

    # download the original pytorch weights
    print(f"[1/6] downloading {model_repo} from huggingface...")
    local_dir = snapshot_download(model_repo)

    turbo_dir = os.path.join(local_dir, turbo_subdir)
    vae_dir = os.path.join(local_dir, vae_subdir)
    text_dir = os.path.join(local_dir, text_encoder_subdir)

    with tempfile.TemporaryDirectory() as out_dir:
        # convert dit + condition encoder
        print("[2/6] converting dit and condition encoder...")
        state_dict = safetensors.torch.load_file(
            os.path.join(turbo_dir, "model.safetensors")
        )

        weights = {}
        for key, value in state_dict.items():
            if not key.startswith("decoder.") and not key.startswith("encoder."):
                continue
            np_val = value.cpu().float().numpy()

            if key.startswith("decoder."):
                new_key = key.replace("decoder.", "dit.")
                if "proj_in.1." in new_key:
                    new_key = new_key.replace("proj_in.1.", "proj_in.")
                    if new_key.endswith(".weight"):
                        np_val = np_val.swapaxes(1, 2)
                elif "proj_out.1." in new_key:
                    new_key = new_key.replace("proj_out.1.", "proj_out.")
                    if new_key.endswith(".weight"):
                        np_val = np_val.transpose(1, 2, 0)
                elif "rotary_emb" in new_key:
                    continue
                weights[new_key] = mx.array(np_val)

            elif key.startswith("encoder."):
                if "rotary_emb" in key:
                    continue
                weights[key] = mx.array(np_val)

        # extract empty timbre token
        print("[3/6] extracting empty timbre token...")
        silence_path = os.path.join(turbo_dir, "silence_latent.pt")
        if os.path.exists(silence_path):
            sys.path.insert(0, local_dir)
            from acestep.models.base.configuration_acestep_v15 import (
                AceStepConfig as PTConfig,
            )
            from acestep.models.base.modeling_acestep_v15_base import (
                AceStepConditionEncoder,
            )

            with open(os.path.join(turbo_dir, "config.json")) as f:
                cfg = json.load(f)
            pt_config = PTConfig(**cfg)
            pt_encoder = AceStepConditionEncoder(pt_config)
            encoder_state = {
                k.replace("encoder.", ""): v
                for k, v in state_dict.items()
                if k.startswith("encoder.")
            }
            pt_encoder.load_state_dict(encoder_state, strict=False)
            pt_encoder.eval()

            pt_silence = torch.load(silence_path, map_location="cpu", weights_only=True)
            if pt_silence.shape[-1] == 128 or pt_silence.shape[-1] == 64:
                silence_input = pt_silence[:, :750, :]
            else:
                silence_input = pt_silence.transpose(1, 2)[:, :750, :]

            with torch.no_grad():
                empty_emb, _ = pt_encoder.timbre_encoder(
                    silence_input, torch.zeros((1,), dtype=torch.long)
                )
            weights["empty_timbre_token"] = mx.array(empty_emb.cpu().numpy())

            # null condition emb
            if "null_condition_emb" in state_dict:
                weights["null_condition_emb"] = mx.array(
                    state_dict["null_condition_emb"].cpu().float().numpy()
                )

            # save silence latent as npy
            np.save(os.path.join(out_dir, "silence_latent.npy"), pt_silence.numpy())

        mx.save_safetensors(os.path.join(out_dir, "model.safetensors"), weights)

        # copy config
        shutil.copy2(os.path.join(turbo_dir, "config.json"), out_dir)

        # convert vae
        print("[4/6] converting vae...")
        pt_vae = AutoencoderOobleck.from_pretrained(vae_dir)
        vae_weights = {}

        def _fuse_weight_norm(weight_g, weight_v, eps=1e-9):
            v_flat = weight_v.reshape(weight_v.shape[0], -1)
            norm = np.linalg.norm(v_flat, axis=1).reshape(weight_g.shape)
            return weight_v * (weight_g / np.maximum(norm, eps))

        vae_state = pt_vae.state_dict()
        processed = set()
        for key in sorted(vae_state.keys()):
            if key in processed:
                continue
            if key.endswith(".weight_g"):
                base = key[: -len(".weight_g")]
                v_key = base + ".weight_v"
                g = vae_state[key].detach().cpu().float().numpy()
                v = vae_state[v_key].detach().cpu().float().numpy()
                w = _fuse_weight_norm(g, v)
                if "conv_t1" in base:
                    w = w.transpose(1, 2, 0)
                else:
                    w = w.swapaxes(1, 2)
                vae_weights[base + ".weight"] = mx.array(w)
                processed.add(key)
                processed.add(v_key)
                continue
            if key.endswith(".weight_v"):
                continue
            val = vae_state[key].detach().cpu().float().numpy()
            if key.endswith(".alpha") or key.endswith(".beta"):
                val = val.squeeze()
            if "conv" in key and key.endswith(".weight"):
                if "conv_t1" in key:
                    val = val.transpose(1, 2, 0)
                else:
                    val = val.swapaxes(1, 2)
            vae_weights[key] = mx.array(val)
            processed.add(key)

        vae_out = os.path.join(out_dir, "vae")
        os.makedirs(vae_out, exist_ok=True)
        mx.save_safetensors(
            os.path.join(vae_out, "diffusion_pytorch_model.safetensors"), vae_weights
        )
        shutil.copy2(os.path.join(vae_dir, "config.json"), vae_out)

        # convert text encoder
        print("[5/6] converting text encoder...")
        if os.path.exists(text_dir):
            text_state = safetensors.torch.load_file(
                os.path.join(text_dir, "model.safetensors")
            )
            text_weights = {}
            for k, v in text_state.items():
                if "q_norm" in k or "k_norm" in k:
                    continue
                clean_key = k[6:] if k.startswith("model.") else k
                text_weights[clean_key] = mx.array(v.cpu().float().numpy())

            mx.save_safetensors(
                os.path.join(out_dir, "text_encoder.safetensors"), text_weights
            )

            # copy tokenizer files
            for fname in [
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
            ]:
                src = os.path.join(text_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, out_dir)

        # upload
        print(f"[6/6] uploading to {hf_repo}...")
        api.create_repo(repo_id=hf_repo, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=out_dir,
            repo_id=hf_repo,
            repo_type="model",
            commit_message="add converted MLX weights for ACE-Step v1.5 turbo",
        )
        print(f"done! uploaded to https://huggingface.co/{hf_repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="ACE-Step/Ace-Step1.5", help="source HF repo"
    )
    parser.add_argument(
        "--hf-repo",
        default="mlx-community/ACE-Step-v1.5-turbo-mlx",
        help="target HF repo",
    )
    args = parser.parse_args()
    convert_and_upload(model_repo=args.model, hf_repo=args.hf_repo)
