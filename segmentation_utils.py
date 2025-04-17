"""
Various utility functions for cleaning the raw predictions from the
segmentation model (mostly reducing storage size, see
visualize-segmentation-predictions.ipynb for details).
"""

import numpy as np


def optimize_rle_encode_mask(mask, compressed=True):
    """
    Compress a boolean segmentation mask using RLE with optimization.
    
    Args:
        mask: 2D boolean numpy array where True indicates the object
        compressed: Whether to use additional compression on the RLE counts
    
    Returns:
        Compressed representation of the mask
    """
    # Run-length encoding (COCO format)
    mask_uint8 = np.asfortranarray(mask.astype(np.uint8))
    from pycocotools import mask as mask_util
    rle = mask_util.encode(mask_uint8)

    # Convert to more compact format
    if compressed:
        # Store as bytes directly instead of ASCII string
        return {
            "counts":
                rle["counts"] if isinstance(rle["counts"], bytes) else
                rle["counts"].encode('ascii'),
            "size":
                rle["size"]
        }
    else:
        return {
            "counts":
                rle["counts"].decode('ascii') if isinstance(
                    rle["counts"], bytes) else rle["counts"],
            "size":
                rle["size"]
        }


def optimize_rle_decode_mask(rle):
    """
    Decompress a mask from its compressed representation.
    
    Args:
        rle: The compressed mask data
    
    Returns:
        Decompressed boolean mask
    """
    from pycocotools import mask as mask_util

    # Handle both string and bytes formats
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts
    else:
        counts = counts.encode('ascii') if isinstance(counts, str) else counts

    rle_data = {"counts": counts, "size": rle["size"]}
    mask = mask_util.decode(rle_data)
    return mask.astype(bool)

from copy import deepcopy


def get_mask_preds(frame_preds):
    """
    Remove bounding boxes from the prediction data structure to save space.
    Bounding boxes can be reconstructed from segmentation masks later if
    needed.
    
    Args:
        preds: Original predictions with bounding boxes
        
    Returns:
        Modified predictions with bounding boxes removed
    """
    return frame_preds.reshape(-1)[1]


def filter_mask_preds_by_confidence(frame_mask_preds, conf_threshold=0.2):
    """
    Filter masks by confidence threshold.
    
    Args:
        preds: Prediction data to filter
        conf_threshold: Only keep predictions with confidence above this
        threshold
        
    Returns:
        Filtered predictions and statistics about filtering
    """
    # Convert from tuple to list
    frame_mask_preds = list(frame_mask_preds)

    filtered_masks = []
    filtered_confs = []

    for class_idx, (class_masks, class_confs) in enumerate(
            zip(frame_mask_preds[0], frame_mask_preds[1])):
        # Filter masks by confidence
        keep_indices = [
            i for i, conf in enumerate(class_confs)
            if conf > conf_threshold
        ]

        # Only keep masks above threshold
        filtered_class_masks = [class_masks[i] for i in keep_indices]
        filtered_class_confs = [class_confs[i] for i in keep_indices]

        filtered_masks.append(filtered_class_masks)
        filtered_confs.append(filtered_class_confs)

        # Update the frame data
    frame_mask_preds = [filtered_masks, filtered_confs]

    return frame_mask_preds


def optimize_mask_encoding(frame_mask_preds, use_binary=True):
    """
    Optimize mask encoding by converting RLE to binary format.
    
    Args:
        preds: Prediction data with masks
        use_binary: Whether to use binary encoding for RLE counts
        
    Returns:
        Predictions with optimized mask encoding
    """
    # Optimize mask encoding
    optimized_masks = []

    for class_masks in frame_mask_preds[0]:
        optimized_class_masks = [
            optimize_rle_encode_mask(mask, compressed=use_binary)
            for mask in class_masks
        ]
        optimized_masks.append(optimized_class_masks)

    # Update the frame data
    frame_mask_preds[0] = optimized_masks

    return frame_mask_preds

def clean_segmentation_preds(frame_preds):
    """
    frame_preds: Raw predictions output by the segmentation model for a single
    frame.
    
    This function:
     1. Removes bounding boxes from the prediction data structure to save
        space.
     2. Filters masks by confidence threshold.
     3. Optimizes mask encoding by converting binary mask to RLE in binary
        format.
    """
    assert isinstance(frame_preds, tuple), type(frame_preds)
    # Subsequent code is written assuming frame_preds is converted to a numpy
    # array.
    frame_preds = np.array(frame_preds)
    mask_preds = get_mask_preds(frame_preds)
    mask_preds = filter_mask_preds_by_confidence(mask_preds)
    mask_preds = optimize_mask_encoding(mask_preds)
    return mask_preds

def save_with_gzip(data, output_file, compression_level=4):
    """
    Save data with gzip compression.
    
    default compression level of 4 was found to be a good compromise between
    compression speed and compression ratio
    (visualize-segmentation-predictions.ipynb).
    
    Args:
        data: Data to save output_file: Path to save the compressed file
        compression_level: gzip compression level (1-9)
        
    Returns:
        File size in MB
    """
    import pickle
    import gzip
    import os
    import time

    output_file = str(output_file)
    assert output_file.endswith(".pl.gz"), output_file

    print(f"Saving with gzip compression level {compression_level}...")
    start_time = time.time()

    with gzip.open(output_file, 'wb', compresslevel=compression_level) as f:
        pickle.dump(data, f)

    elapsed_time = time.time() - start_time
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

    print(f"Saved to {output_file}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Saving time: {elapsed_time:.2f} seconds")

    return file_size_mb


def load_from_gzip(input_file):
    """
    Load data from a gzipped pickle file.
    
    Args:
        input_file: Path to the compressed file
        
    Returns:
        Loaded data
    """
    import pickle
    import gzip
    import time

    print(f"Loading from {input_file}...")
    start_time = time.time()

    with gzip.open(input_file, 'rb') as f:
        data = pickle.load(f)

    elapsed_time = time.time() - start_time

    print(f"Loaded {len(data)} items")
    print(f"Loading time: {elapsed_time:.2f} seconds")

    return data