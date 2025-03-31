import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from transformers import GPT2Model, GPT2Config, AutoImageProcessor
from PIL import Image


class GPT2VisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        
        # Define image size and patch size to mimic CLIP's behavior
        self._image_size = 224
        self._patch_size = 16
        self._num_patches_per_side = self._image_size // self._patch_size
        self._num_patches = self._num_patches_per_side ** 2
        
        # GPT2 specific configuration
        self.gpt2_hidden_size = 1024 if "medium" in self.vision_tower_name else 768  # Default to medium

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = GPT2Config.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        print(f"Loading GPT2VisionTower from {self.vision_tower_name}")
        # Use CLIP's image processor for consistent preprocessing
        self.image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        # Load GPT2 model
        self.gpt2_config = GPT2Config.from_pretrained(self.vision_tower_name)
        self.vision_tower = GPT2Model.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        
        # Add patch embedding layer to convert image patches to embeddings
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.gpt2_hidden_size,
            kernel_size=self._patch_size,
            stride=self._patch_size,
            bias=False
        )
        
        # Add position embeddings for patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self._num_patches, self.gpt2_hidden_size)
        )
        # Initialize position embeddings with sinusoidal pattern
        self._init_pos_embedding()
        
        self.is_loaded = True

    def _init_pos_embedding(self):
        # Initialize position embeddings with sinusoidal pattern (similar to Transformer)
        position = torch.arange(0, self._num_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.gpt2_hidden_size, 2) * -(np.log(10000.0) / self.gpt2_hidden_size))
        pos_embed = torch.zeros(1, self._num_patches, self.gpt2_hidden_size)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)
        self.pos_embed.data.copy_(pos_embed)

    def feature_select(self, image_forward_outs):
        # Similar to CLIP, select features from specific layer
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            # For GPT2, we take all token representations
            image_features = image_features
        elif self.select_feature == 'cls_patch':
            # Add dummy first token for cls (since GPT2 doesn't have cls token)
            batch_size = image_features.shape[0]
            cls_embedding = torch.zeros(
                batch_size, 1, self.gpt2_hidden_size, 
                device=image_features.device, 
                dtype=image_features.dtype
            )
            image_features = torch.cat([cls_embedding, image_features], dim=1)
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def _process_image(self, image):
        # Convert image to patches and project to embedding space
        if len(image.shape) == 3:  # Add batch dimension if needed
            image = image.unsqueeze(0)
        
        # Process image through patch embedding layer
        # Rearrange from [B, C, H, W] to [B, C, H, W]
        patches = self.patch_embed(image)
        # Rearrange from [B, E, H', W'] to [B, H'*W', E]
        patches = patches.flatten(2).transpose(1, 2)
        
        # Add position embeddings
        patches = patches + self.pos_embed
        
        return patches

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                # Process image to get patch embeddings
                patches = self._process_image(image.to(device=self.device, dtype=self.dtype))
                
                # Pass through GPT2
                outputs = self.vision_tower(
                    inputs_embeds=patches,
                    output_hidden_states=True
                )
                
                # Select features as in CLIP
                image_feature = self.feature_select(outputs).to(image.dtype)
                image_features.append(image_feature)
        else:
            # Process batch of images to get patch embeddings
            patches = self._process_image(images.to(device=self.device, dtype=self.dtype))
            
            # Pass through GPT2
            outputs = self.vision_tower(
                inputs_embeds=patches,
                output_hidden_states=True
            )
            
            # Select features
            image_features = self.feature_select(outputs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype if self.is_loaded else torch.float32

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device if self.is_loaded else torch.device('cpu')

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.gpt2_hidden_size

    @property
    def num_patches_per_side(self):
        return self._num_patches_per_side

    @property
    def num_patches(self):
        return self._num_patches 