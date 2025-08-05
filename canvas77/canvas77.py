"""
Canvas-77 Node - Fixed Version
File: canvas_node.py
Directory: ComfyUI/custom_nodes/Alo77/canvas77/
"""

import numpy as np
import torch
from PIL import Image
import json


class Canvas77:
    """
    Advanced canvas composer for layering multiple images with precise control.
    Perfect companion for Outline-77 node outputs.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE", {
                    "tooltip": "Background image for the canvas"
                }),
                "canvas_width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "display": "slider",
                    "tooltip": "Canvas width in pixels"
                }),
                "canvas_height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "display": "slider",
                    "tooltip": "Canvas height in pixels"
                }),
                "layer_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Number of image layers to composite"
                }),
                "layer_data": ("STRING", {
                    "default": '{"1":{"x":0,"y":0,"scale":1.0,"rotation":0,"opacity":1.0}}',
                    "multiline": True,
                    "tooltip": "JSON data for layer positioning and properties"
                }),
            },
            "optional": {
                "layer_1": ("IMAGE", {"tooltip": "Layer 1 image"}),
                "layer_2": ("IMAGE", {"tooltip": "Layer 2 image"}),
                "layer_3": ("IMAGE", {"tooltip": "Layer 3 image"}),
                "layer_4": ("IMAGE", {"tooltip": "Layer 4 image"}),
                "layer_5": ("IMAGE", {"tooltip": "Layer 5 image"}),
                "layer_6": ("IMAGE", {"tooltip": "Layer 6 image"}),
                "layer_7": ("IMAGE", {"tooltip": "Layer 7 image"}),
                "layer_8": ("IMAGE", {"tooltip": "Layer 8 image"}),
                "layer_9": ("IMAGE", {"tooltip": "Layer 9 image"}),
                "layer_10": ("IMAGE", {"tooltip": "Layer 10 image"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composed_image",)
    FUNCTION = "compose_canvas"
    CATEGORY = "Alo77/Canvas"
    OUTPUT_NODE = False
    
    def tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        np_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        if np_array.shape[2] == 1:
            np_array = np.repeat(np_array, 3, axis=2)
            return Image.fromarray(np_array, 'RGB')
        elif np_array.shape[2] == 3:
            return Image.fromarray(np_array, 'RGB')
        elif np_array.shape[2] == 4:
            return Image.fromarray(np_array, 'RGBA')
        else:
            return Image.fromarray(np_array[:, :, :3], 'RGB')
    
    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor."""
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')
        
        np_array = np.array(pil_image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np_array).unsqueeze(0)
        return tensor
    
    def parse_layer_data(self, layer_data_str, layer_count):
        """Parse JSON layer data with fallbacks."""
        try:
            layer_data = json.loads(layer_data_str)
        except:
            # Fallback to default data
            layer_data = {}
        
        # Ensure all layers have default values
        for i in range(1, layer_count + 1):
            if str(i) not in layer_data:
                layer_data[str(i)] = {
                    "x": 0,
                    "y": 0,
                    "scale": 1.0,
                    "rotation": 0,
                    "opacity": 1.0
                }
            else:
                # Fill missing properties with defaults
                defaults = {"x": 0, "y": 0, "scale": 1.0, "rotation": 0, "opacity": 1.0}
                for key, default_val in defaults.items():
                    if key not in layer_data[str(i)]:
                        layer_data[str(i)][key] = default_val
        
        return layer_data
    
    def transform_image(self, pil_image, scale, rotation, opacity):
        """Apply transformations to an image."""
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')
        
        # Apply scaling
        if scale != 1.0:
            new_width = max(1, int(pil_image.width * scale))
            new_height = max(1, int(pil_image.height * scale))
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Apply rotation
        if rotation != 0.0:
            pil_image = pil_image.rotate(rotation, expand=True, fillcolor=(0, 0, 0, 0))
        
        # Apply opacity
        if opacity != 1.0:
            img_array = np.array(pil_image, dtype=np.float32)
            img_array[:, :, 3] *= opacity
            pil_image = Image.fromarray(img_array.astype(np.uint8), 'RGBA')
        
        return pil_image
    
    def composite_layer(self, canvas, layer_image, x, y):
        """Composite a layer onto the canvas at specified position."""
        if layer_image is None:
            return canvas
        
        if canvas.mode != 'RGBA':
            canvas = canvas.convert('RGBA')
        if layer_image.mode != 'RGBA':
            layer_image = layer_image.convert('RGBA')
        
        # Create temporary canvas for