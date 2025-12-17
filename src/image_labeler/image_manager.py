import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import readlif.reader
from PIL import Image

def get_image_folder():
    return "images"

"""
Loads a red scaled PNG
"""
def load_png(image_name):
    folder = get_image_folder()
    path = os.path.join(folder, image_name)

    img = Image.open(path)

    arr = np.array(img)

    return arr

"""
LIF Maximum Intensity Projection Tool

Reads Leica Image Format (LIF) files and creates maximum intensity projections
across Z-slices. Handles mosaic/tiled images and provides clean output options.

Usage:
    python lif_max_projection.py <lif_file> <output_file> [options]
    python lif_max_projection.py <lif_file> --info-only

Source: https://github.com/RADICAL-UBC/neurites/blob/main/lif_max_projection.py
"""
def load_lif_info(lif_file_path):
    """
    Load and display information about a LIF file.

    Args:
        lif_file_path (str): Path to the LIF file

    Returns:
        readlif.reader.LifFile: The loaded LIF file object
    """
    print(f"Loading LIF file: {lif_file_path}")
    lif = readlif.reader.LifFile(lif_file_path)

    print(f"\n{'=' * 60}")
    print(f"LIF File Information")
    print(f"{'=' * 60}")
    print(f"Number of images in file: {lif.num_images}")

    for i in range(lif.num_images):
        image = lif.get_image(i)
        is_mosaic = image.dims.m > 1

        print(f"\nImage {i}:")
        print(f"  Name: {image.name}")
        print(f"  Dimensions: {image.dims}")
        print(f"  Channels: {image.channels}")
        print(f"  Time points: {image.nt}")
        print(f"  Z-slices: {image.nz}")
        print(f"  Mosaic tiles: {image.dims.m}")
        print(f"  Bit depth: {image.bit_depth}")

        if is_mosaic:
            print(f"  ⚠️  WARNING: This is a MOSAIC image with {image.dims.m} tiles")
            print(f"      Processing individual tiles - may need stitching!")
            # Check if there's a merged version
            if i + 1 < lif.num_images:
                next_image = lif.get_image(i + 1)
                if "_Merged" in next_image.name or "Merged" in next_image.name:
                    print(f"      💡 TIP: Image {i + 1} ('{next_image.name}') appears to be the merged version")

    print(f"{'=' * 60}\n")
    return lif


def create_max_projection(lif_file_path, image_index=0, channel=0, time=0, allow_mosaic=False):
    """
    Create a maximum intensity projection from a LIF file.

    Args:
        lif_file_path (str): Path to the LIF file
        image_index (int): Index of the image to process
        channel (int): Channel index
        time (int): Time point index
        allow_mosaic (bool): Skip mosaic warning prompt

    Returns:
        tuple: (max_projection_array, single_slice_array, image_metadata)
    """
    # Load the LIF file
    lif = readlif.reader.LifFile(lif_file_path)

    if image_index >= lif.num_images:
        raise ValueError(f"Image index {image_index} out of range. File has {lif.num_images} images.")

    image = lif.get_image(image_index)

    # Check if this is a mosaic image
    is_mosaic = image.dims.m > 1
    if is_mosaic:
        print(f"\n{'!' * 60}")
        print(f"WARNING: Image {image_index} is a MOSAIC with {image.dims.m} tiles!")
        print(f"{'!' * 60}")
        print(f"This image contains {image.dims.m} separate tiles that may need stitching.")
        print(f"Each tile is {image.dims.x} x {image.dims.y} pixels.")
        print(f"\nOptions:")
        print(f"  1. Look for a '_Merged' version in the file (use --info-only)")
        print(f"  2. Continue processing individual tiles (may look incomplete)")
        print(f"  3. Use specialized mosaic stitching tools")

        # Check for merged version
        for i in range(lif.num_images):
            if i != image_index:
                other = lif.get_image(i)
                if "_Merged" in other.name or "Merged" in other.name:
                    if image.name.replace("_Merged", "").replace(" _Merged", "") in other.name or \
                            other.name.replace("_Merged", "").replace(" _Merged", "") in image.name:
                        print(f"\n💡 FOUND: Image {i} ('{other.name}') appears to be the merged version!")
                        print(f"   Recommended: Use --image {i} instead")

        print(f"{'!' * 60}\n")

        if not allow_mosaic:
            response = input("Continue with mosaic tiles anyway? [y/N]: ").strip().lower()
            if response not in ['y', 'yes']:
                raise ValueError("Processing cancelled. Please select a different image.")
        else:
            print("(Skipping prompt - processing mosaic as-is)")

    # Validate parameters
    if channel >= image.channels:
        raise ValueError(f"Channel {channel} out of range. Image has {image.channels} channels.")

    if time >= image.nt:
        raise ValueError(f"Time {time} out of range. Image has {image.nt} time points.")

    print(f"Processing image: {image.name}")
    print(f"  Channel: {channel}")
    print(f"  Time: {time}")
    print(f"  Z-slices: {image.nz}")
    if is_mosaic:
        print(f"  Mosaic tiles: {image.dims.m} (processing first tile only)")

    # Load all Z-slices
    print(f"\nLoading {image.nz} Z-slices...")
    z_stack = []

    for z in range(image.nz):
        frame = image.get_frame(z=z, t=time, c=channel)
        z_stack.append(np.array(frame))

        # Progress indicator
        if image.nz > 10 and (z + 1) % 10 == 0:
            print(f"  Progress: {z + 1}/{image.nz} slices loaded")

    print(f"  Complete: {image.nz}/{image.nz} slices loaded")

    # Convert to numpy array
    z_stack = np.array(z_stack)

    # Create maximum intensity projection
    print(f"\nCreating maximum intensity projection...")
    max_projection = np.max(z_stack, axis=0)

    # Get middle slice for comparison
    middle_idx = image.nz // 2
    single_slice = z_stack[middle_idx]

    # Metadata
    metadata = {
        'name': image.name,
        'num_slices': image.nz,
        'channel': channel,
        'time': time,
        'dimensions': image.dims,
        'shape': max_projection.shape,
        'intensity_min': max_projection.min(),
        'intensity_max': max_projection.max(),
        'middle_slice_index': middle_idx
    }

    print(f"  Shape: {max_projection.shape}")
    print(f"  Intensity range: {metadata['intensity_min']} to {metadata['intensity_max']}")

    return max_projection, single_slice, metadata