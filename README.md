# SDXL IRs and Scripts


## Model IRs and weights

> [!CAUTION]
> IRs in the following table might be stale. Use the ones in the
> `base_ir/` directory instead.

> [!NOTE]
> SDXL-turbo is only different from SDXL in its usage and training/weights.
> The model architecture (and therefore the weights-stripped MLIR) are equivalent.

Variant | Submodel | MLIR (No Weights) (Config A) | safetensors | Splat IRPA | MLIR (No Weights) (Config B)
:-----: | :---------: | :-------------: | :-------------: | :------: | :--------:
| SDXL1.0 1024x1024 (f16, BS1, len64) |
|  | UNet + attn | [Torch](https://sharkpublic.blob.core.windows.net/sharkpublic/ian/stable_diffusion_xl_base_1_0_64_1024x1024_fp16_unet_torchir_nithin.mlir) - [Linalg](https://sharkpublic.blob.core.windows.net/sharkpublic/ian/stable_diffusion_xl_base_1_0_64_1024x1024_fp16_unet_linalg_nithin.mlir)| - | - | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_b/fp16/stable_diffusion_xl_base_1_0_64_1024x1024_fp16_unet.mlir)
|  | UNet + PNDMScheduler | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/stable_diffusion_xl_base_1_0_scheduler_rocm.mlir)
|  | Clip1 | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_a/fp16/stable_diffusion_xl_base_1_0_64_fp16_clip_1.mlir) | - | - |
|  | Clip2 | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_a/fp16/stable_diffusion_xl_base_1_0_64_fp16_clip_2.mlir) | - | - |
|  | VAE decode + attn | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_a/fp16/stable_diffusion_xl_base_1_0_1024x1024_fp16_vae_decode.mlir) | - | = | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_b/fp16/stable_diffusion_xl_base_1_0_1024x1024_fp16_vae_decode.mlir)
|  | VAE encode + attn | [GCloud][sdxl-1-1024x1024-f16-stripped-weight-vae-encode] | Same as decode | - | -
| SDXL1.0 1024x1024 (f32, BS1, len64) |
|  | UNet + attn | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_a/fp32/stable_diffusion_xl_base_1_0_64_1024x1024_fp32_unet.mlir) | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_a/fp32/stable_diffusion_xl_base_1_0_fp32_unet.safetensors) | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_a/fp32/stable_diffusion_xl_base_1_0_fp32_unet.irpa) | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_b/fp32/stable_diffusion_xl_base_1_0_64_1024x1024_fp32_unet.mlir)
|  | Clip1 | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_a/fp32/stable_diffusion_xl_base_1_0_64_fp32_clip_1.mlir) | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_a/fp32/stable_diffusion_xl_base_1_0_fp32_clip_1.safetensors) | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_a/fp32/stable_diffusion_xl_base_1_0_fp32_clip_1.irpa) | -
|  | Clip2 | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_a/fp32/stable_diffusion_xl_base_1_0_64_fp32_clip_2.mlir) | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_a/fp32/stable_diffusion_xl_base_1_0_fp32_clip_2.safetensors) | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_a/fp32/stable_diffusion_xl_base_1_0_fp32_clip_2.irpa) | -
|  | VAE decode + attn | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_a/fp32/stable_diffusion_xl_base_1_0_1024x1024_fp32_vae_decode.mlir) | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_a/fp32/stable_diffusion_xl_base_1_0_fp32_vae_decode.safetensors) | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_a/fp32/stable_diffusion_xl_base_1_0_fp32_vae_decode.irpa) | [Azure](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/cfg_b/fp32/stable_diffusion_xl_base_1_0_1024x1024_fp32_vae_decode.mlir)
| SDXL compiled pipeline IRPAs (f16) |
|  | UNet  |  [scheduled_unet_f16.irpa](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/SDXL_weights_fp16/scheduled_unet_fp16.irpa)
|  | Prompt Encoder (CLIP1 + CLIP2) | [prompt_encoder_f16.irpa](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/SDXL_weights_fp16/prompt_encoder_fp16.irpa)
|  | VAE |  [vae_decode_f16.irpa](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/SDXL_weights_fp16/vae_decode_fp16.irpa)
