import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .gpt2_vision_encoder import GPT2VisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    use_gpt2 = getattr(vision_tower_cfg, 'use_gpt2_vision', False)
    
    # If GPT2 is explicitly requested
    if use_gpt2 or "gpt2" in vision_tower:
        return GPT2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
