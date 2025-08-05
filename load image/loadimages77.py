"""
LoadImages-77 Node - Fixed Version
File: load_node.py
Directory: ComfyUI/custom_nodes/Alo77/loadimages77/
"""

import os
import numpy as np
import torch
from PIL import Image, ImageOps, ExifTags
import folder_paths
import hashlib


class LoadImages77:
    """
    Advanced image loader with batch processing, filtering, and auto-processing features.
    Perfect for feeding multiple images to Canvas-77 and Outline-77 nodes.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.exists(input_dir):
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif'))]
        
        return {
            "required": {
                "directory_mode": (["single_file", "batch_folder", "filtered_batch"], {
                    "default": "single_file",
                    "tooltip": "Loading mode: single file, all files in folder, or filtered batch"
                }),
                "image": (sorted(image_files) if image_files else ["no_images_found"], {
                    "tooltip": "Select image file (single_file mode)"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Number of images to load in batch mode"
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Starting index for batch loading"
                }),
                "auto_orient": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically orient images based on EXIF data"
                }),
                "resize_mode": (["none", "fit_canvas", "fill_canvas", "stretch"], {
                    "default": "none",
                    "tooltip": "Resize mode for loaded images"
                }),
                "target_width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                    "display": "slider",
                    "tooltip": "Target width for resizing"
                }),
                "target_height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                    "display": "slider",
                    "tooltip": "Target height for resizing"
                }),
            },
            "optional": {
                "file_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Filter files by name (supports wildcards: *.png, *logo*, etc.)"
                }),
                "sort_by": (["name", "date", "size", "random"], {
                    "default": "name",
                    "tooltip": "Sort order for batch loading"
                }),
                "include_subfolders": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include images from subfolders"
                }),
                "output_format": (["original", "RGB", "RGBA"], {
                    "default": "original",
                    "tooltip": "Force output format"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "filenames", "count")
    FUNCTION = "load_images"
    CATEGORY = "Alo77/Image"
    OUTPUT_NODE = False
    
    def apply_exif_orientation(self, image):
        """Apply EXIF orientation data to rotate image correctly."""
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            
            exif = image._getexif()
            if exif is not None:
                orientation_value = exif.get(orientation)
                if orientation_value == 3:
                    image = image.rotate(180, expand=True)
                elif orientation_value == 6:
                    image = image.rotate(270, expand=True)
                elif orientation_value == 8:
                    image = image.rotate(90, expand=True)
        except (AttributeError, KeyError, TypeError):
            pass
        return image
    
    def resize_image(self, image, mode, target_width, target_height):
        """Resize image according to specified mode."""
        if mode == "none":
            return image
        
        original_width, original_height = image.size
        
        if mode == "fit_canvas":
            # Fit image within canvas while maintaining aspect ratio
            image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
            # Create canvas and center image
            canvas = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))
            paste_x = (target_width - image.width) // 2
            paste_y = (target_height - image.height) // 2
            if image.mode == 'RGBA':
                canvas.paste(image, (paste_x, paste_y), image)
            else:
                canvas.paste(image, (paste_x, paste_y))
            return canvas
        
        elif mode == "fill_canvas":
            # Fill canvas while maintaining aspect ratio (crop if needed)
            ratio = max(target_width / original_width, target_height / original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Crop to target size
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            return image.crop((left, top, right, bottom))
        
        elif mode == "stretch":
            # Stretch to exact dimensions (may distort)
            return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        return image
    
    def filter_files(self, files, file_filter):
        """Filter files based on pattern."""
        if not file_filter:
            return files
        
        import fnmatch
        filtered = []
        
        # Split multiple patterns by comma
        patterns = [p.strip() for p in file_filter.split(',')]
        
        for file in files:
            for pattern in patterns:
                if fnmatch.fnmatch(file.lower(), pattern.lower()):
                    filtered.append(file)
                    break
        
        return filtered
    
    def get_file_list(self, directory_mode, file_filter, include_subfolders, sort_by):
        """Get list of files based on parameters."""
        input_dir = folder_paths.get_input_directory()
        files = []
        
        if not os.path.exists(input_dir):
            return []
        
        if include_subfolders:
            for root, dirs, filenames in os.walk(input_dir):
                for filename in filenames:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif')):
                        rel_path = os.path.relpath(os.path.join(root, filename), input_dir)
                        files.append(rel_path)
        else:
            files = [f for f in os.listdir(input_dir) 
                    if os.path.isfile(os.path.join(input_dir, f)) and 
                    f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif'))]
        
        # Apply filter
        files = self.filter_files(files, file_filter)
        
        # Sort files
        if sort_by == "name":
            files.sort()
        elif sort_by == "date":
            files.sort(key=lambda x: os.path.getmtime(os.path.join(input_dir, x)))
        elif sort_by == "size":
            files.sort(key=lambda x: os.path.getsize(os.path.join(input_dir, x)))
        elif sort_by == "random":
            import random
            random.shuffle(files)
        
        return files
    
    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor."""
        # Ensure consistent format
        if pil_image.mode not in ['RGB', 'RGBA']:
            if pil_image.mode == 'L':
                pil_image = pil_image.convert('RGB')
            else:
                pil_image = pil_image.convert('RGBA')
        
        np_array = np.array(pil_image, dtype=np.float32) / 255.0
        
        # Ensure 3 or 4 channels
        if len(np_array.shape) == 2:
            np_array = np.stack([np_array] * 3, axis=-1)
        
        tensor = torch.from_numpy(np_array)
        return tensor
    
    def load_images(self, directory_mode, image, batch_size, start_index, auto_orient, 
                   resize_mode, target_width, target_height, file_filter="", 
                   sort_by="name", include_subfolders=False, output_format="original"):
        """Main image loading function."""
        try:
            input_dir = folder_paths.get_input_directory()
            loaded_images = []
            filenames = []
            
            if not os.path.exists(input_dir):
                fallback = torch.zeros((1, target_height, target_width, 3), dtype=torch.float32)
                return (fallback, ["no_input_directory"], 0)
            
            if directory_mode == "single_file":
                # Load single file
                if image == "no_images_found":
                    fallback = torch.zeros((1, target_height, target_width, 3), dtype=torch.float32)
                    return (fallback, ["no_images_found"], 0)
                    
                image_path = os.path.join(input_dir, image)
                if os.path.exists(image_path):
                    try:
                        pil_image = Image.open(image_path)
                        
                        if auto_orient:
                            pil_image = self.apply_exif_orientation(pil_image)
                        
                        # Apply output format
                        if output_format == "RGB":
                            pil_image = pil_image.convert('RGB')
                        elif output_format == "RGBA":
                            pil_image = pil_image.convert('RGBA')
                        
                        # Resize if needed
                        pil_image = self.resize_image(pil_image, resize_mode, target_width, target_height)
                        
                        tensor = self.pil_to_tensor(pil_image)
                        loaded_images.append(tensor)
                        filenames.append(image)
                        
                    except Exception as e:
                        print(f"Error loading {image}: {e}")
            
            else:
                # Batch loading
                files = self.get_file_list(directory_mode, file_filter, include_subfolders, sort_by)
                
                if not files:
                    fallback = torch.zeros((1, target_height, target_width, 3), dtype=torch.float32)
                    return (fallback, ["no_files_found"], 0)
                
                # Apply batch parameters
                end_index = min(start_index + batch_size, len(files))
                selected_files = files[start_index:end_index]
                
                for filename in selected_files:
                    image_path = os.path.join(input_dir, filename)
                    try:
                        pil_image = Image.open(image_path)
                        
                        if auto_orient:
                            pil_image = self.apply_exif_orientation(pil_image)
                        
                        # Apply output format
                        if output_format == "RGB":
                            pil_image = pil_image.convert('RGB')
                        elif output_format == "RGBA":
                            pil_image = pil_image.convert('RGBA')
                        
                        # Resize if needed
                        pil_image = self.resize_image(pil_image, resize_mode, target_width, target_height)
                        
                        tensor = self.pil_to_tensor(pil_image)
                        loaded_images.append(tensor)
                        filenames.append(filename)
                        
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
                        continue
            
            if not loaded_images:
                # Return empty fallback
                fallback = torch.zeros((1, target_height, target_width, 3), dtype=torch.float32)
                return (fallback, ["no_valid_images"], 0)
            
            # Stack images into batch tensor
            if len(loaded_images) == 1:
                batch_tensor = loaded_images[0].unsqueeze(0)
            else:
                # Ensure all images have same dimensions for batching
                target_shape = loaded_images[0].shape
                valid_images = []
                valid_filenames = []
                
                for i, img_tensor in enumerate(loaded_images):
                    if img_tensor.shape == target_shape:
                        valid_images.append(img_tensor)
                        valid_filenames.append(filenames[i])
                
                if valid_images:
                    batch_tensor = torch.stack(valid_images, dim=0)
                    filenames = valid_filenames
                else:
                    batch_tensor = loaded_images[0].unsqueeze(0)
                    filenames = [filenames[0]]
            
            return (batch_tensor, filenames, len(filenames))
            
        except Exception as e:
            print(f"LoadImages77 Error: {str(e)}")
            fallback = torch.zeros((1, target_height, target_width, 3), dtype=torch.float32)
            return (fallback, [f"error: {str(e)}"], 0)


class LoadImages77Simple:
    """
    Simplified version for quick single/batch image loading.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.exists(input_dir):
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        
        return {
            "required": {
                "image": (sorted(image_files) if image_files else ["no_images_found"], {"tooltip": "Select image file"}),
                "load_mode": (["single", "batch_5", "batch_10", "batch_all"], {
                    "default": "single",
                    "tooltip": "Loading mode"
                }),
                "auto_resize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Auto-resize to 1024x1024"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    FUNCTION = "load_simple"
    CATEGORY = "Alo77/Image"
    OUTPUT_NODE = False
    
    def pil_to_tensor(self, pil_image):
        """Convert PIL to tensor."""
        if pil_image.mode not in ['RGB', 'RGBA']:
            pil_image = pil_image.convert('RGB')
        
        np_array = np.array(pil_image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np_array)
        return tensor
    
    def load_simple(self, image, load_mode, auto_resize):
        """Simple loading function."""
        try:
            input_dir = folder_paths.get_input_directory()
            loaded_images = []
            
            if not os.path.exists(input_dir):
                fallback = torch.zeros((1, 1024, 1024, 3), dtype=torch.float32)
                return (fallback, 0)
            
            if image == "no_images_found":
                fallback = torch.zeros((1, 1024, 1024, 3), dtype=torch.float32)
                return (fallback, 0)
            
            if load_mode == "single":
                # Load single image
                image_path = os.path.join(input_dir, image)
                if os.path.exists(image_path):
                    pil_image = Image.open(image_path)
                    
                    if auto_resize:
                        pil_image = pil_image.resize((1024, 1024), Image.Resampling.LANCZOS)
                    
                    tensor = self.pil_to_tensor(pil_image)
                    loaded_images.append(tensor)
            
            else:
                # Batch loading
                files = [f for f in os.listdir(input_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
                files.sort()
                
                if not files:
                    fallback = torch.zeros((1, 1024, 1024, 3), dtype=torch.float32)
                    return (fallback, 0)
                
                # Determine batch size
                if load_mode == "batch_5":
                    batch_size = 5
                elif load_mode == "batch_10":
                    batch_size = 10
                else:  # batch_all
                    batch_size = len(files)
                
                # Find starting index based on selected image
                try:
                    start_idx = files.index(image)
                except ValueError:
                    start_idx = 0
                
                # Load batch
                for i in range(min(batch_size, len(files) - start_idx)):
                    filename = files[start_idx + i]
                    image_path = os.path.join(input_dir, filename)
                    
                    try:
                        pil_image = Image.open(image_path)
                        
                        if auto_resize:
                            pil_image = pil_image.resize((1024, 1024), Image.Resampling.LANCZOS)
                        
                        tensor = self.pil_to_tensor(pil_image)
                        loaded_images.append(tensor)
                    except:
                        continue
            
            if not loaded_images:
                fallback = torch.zeros((1, 1024, 1024, 3), dtype=torch.float32)
                return (fallback, 0)
            
            # Stack into batch
            if len(loaded_images) == 1:
                batch_tensor = loaded_images[0].unsqueeze(0)
            else:
                batch_tensor = torch.stack(loaded_images, dim=0)
            
            return (batch_tensor, len(loaded_images))
            
        except Exception as e:
            print(f"LoadImages77Simple Error: {str(e)}")
            fallback = torch.zeros((1, 1024, 1024, 3), dtype=torch.float32)
            return (fallback, 0)


# Node Registration
NODE_CLASS_MAPPINGS = {
    "LoadImages-77": LoadImages77,
    "LoadImages-77-Simple": LoadImages77Simple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImages-77": "LoadImages-77 📁",
    "LoadImages-77-Simple": "LoadImages-77 Simple 📷"
}