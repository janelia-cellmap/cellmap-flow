#!/usr/bin/env python3
"""
Create bounding boxes interactively in Neuroglancer using Ctrl+Click
Automatically closes when target number of boxes is reached.

Can be used as a function or standalone script.
"""

import argparse
import json
import time
import neuroglancer
import neuroglancer.cli
from pathlib import Path
from typing import List, Dict, Any

neuroglancer.set_server_bind_address("0.0.0.0")


def create_bounding_boxes(
    image_path: str,
    num_boxes: int,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Create bounding boxes interactively in Neuroglancer.
    
    Args:
        image_path: URL or path to image (zarr/n5/precomputed)
        num_boxes: Target number of bounding boxes to create
        verbose: Print progress messages
        
    Returns:
        List of bounding boxes as dicts with keys:
        - name: str (roi_1, roi_2, etc.)
        - offset: list of 3 ints [z, y, x]
        - shape: list of 3 ints [z, y, x]
        - color: str (red)
        - description: str
        
    Example:
        >>> bboxes = create_bounding_boxes(
        ...     "https://example.com/image.zarr",
        ...     num_boxes=3
        ... )
        >>> print(bboxes)
        [{"name": "roi_1", "offset": [100, 200, 300], ...}, ...]
    """
    
    if num_boxes <= 0:
        raise ValueError("num_boxes must be positive")
    
    if verbose:
        print(f"\n✅ Target: {num_boxes} bounding box(es)")
        print("Starting Neuroglancer...\n")
    
    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        # Set coordinate space
        s.dimensions = neuroglancer.CoordinateSpace(
            names=["z", "y", "x"],
            units="nm",
            scales=[8, 8, 8],
        )
        
        # Add image layer
        s.layers["fibsem"] = neuroglancer.ImageLayer(
            source=f"zarr2://{image_path}",
            shader="""#uicontrol invlerp normalized(range=[-1, 1], window=[-1, 1]);
#uicontrol vec3 color color(default="white");
void main() {
  emitRGB(color * normalized());
}""",
        )
        
        # Add LOCAL annotation layer for bounding boxes
        s.layers["annotations"] = neuroglancer.LocalAnnotationLayer(
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units="nm",
                scales=[1, 1, 1],
            ),
        )

    if verbose:
        print("=" * 70)
        print("Neuroglancer Bounding Box Creator (Ctrl+Click to Draw)")
        print("=" * 70)
        print(f"URL: {viewer}")
        print("\nHOW TO CREATE BOUNDING BOXES:")
        print("  1. Open the Neuroglancer URL above")
        print("  2. Click on 'annotations' layer in LEFT PANEL")
        print("  3. Click the BOUNDING BOX TOOL icon (□ symbol)")
        print("  4. Hold CTRL and CLICK to set first corner")
        print("  5. Hold CTRL and CLICK again to set opposite corner")
        print(f"\n🎯 TARGET: {num_boxes} box(es)")
        print("⏱️  Will auto-close when done\n")
        print("=" * 70 + "\n")

    # Monitor annotations and auto-close when target is reached
    last_count = 0
    check_interval = 1  # Check every 1 second
    
    while True:
        try:
            with viewer.txn() as s:
                try:
                    annotations_layer = s.layers["annotations"]
                    if annotations_layer:
                        current_count = len(annotations_layer.annotations)
                        
                        # Print status updates
                        if current_count != last_count:
                            if verbose:
                                print(f"📦 Boxes created: {current_count}/{num_boxes}")
                            last_count = current_count
                            
                            # Check if target reached
                            if current_count >= num_boxes:
                                if verbose:
                                    print(f"\n✅ Target reached! Extracting {num_boxes} bounding box(es)...\n")
                                break
                except (KeyError, IndexError):
                    # Layer doesn't exist yet or no annotations
                    pass
            
            time.sleep(check_interval)
        
        except KeyboardInterrupt:
            if verbose:
                print("\n\n⚠️  Cancelled by user. Extracting current boxes...\n")
            break

    # Extract annotations from viewer state
    with viewer.txn() as s:
        viewer_state = s.to_json()
    
    viewer_json = viewer_state
    
    # Get the annotations layer
    annotations_layer = None
    for layer in viewer_json.get("layers", []):
        if layer.get("name") == "annotations":
            annotations_layer = layer
            break
    
    bboxes_list = []
    
    if annotations_layer and "annotations" in annotations_layer:
        annotations = annotations_layer["annotations"]
        
        for i, ann in enumerate(annotations):
            if ann.get("type") == "axis_aligned_bounding_box":
                point_a = ann["pointA"]
                point_b = ann["pointB"]
                
                # Ensure point_a is the min and point_b is the max
                offset = [min(point_a[j], point_b[j]) for j in range(3)]
                max_point = [max(point_a[j], point_b[j]) for j in range(3)]
                shape = [int(max_point[j] - offset[j]) for j in range(3)]
                offset = [int(x) for x in offset]
                
                bbox_dict = {
                    "offset": offset,
                    "shape": shape,
                }
                bboxes_list.append(bbox_dict)
        
        if verbose:
            print("=" * 70)
            print("EXTRACTED BOUNDING BOXES")
            print("=" * 70)
            print("\nbounding_boxes = [")
            
            for bbox in bboxes_list:
                print(f"    {{")
                print(f'        "offset": {bbox["offset"]},')
                print(f'        "shape": {bbox["shape"]},')
                print(f"    }},")
            
            print("]")
            print("\n" + "=" * 70)
            print(f"Total bounding boxes: {len(bboxes_list)}")
            print("=" * 70 + "\n")
    
    return bboxes_list


if __name__ == "__main__":
    # Command-line interface
    parser = argparse.ArgumentParser(
        description="Create bounding boxes in Neuroglancer and export as JSON"
    )
    parser.add_argument(
        "--image-path",
        default="https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8/",
        help="Path or URL to image (default: jrc_mus-salivary-1)"
    )
    parser.add_argument(
        "--num-boxes",
        type=int,
        default=None,
        help="Number of boxes to create (interactive if not specified)"
    )
    parser.add_argument(
        "--output",
        default="exported_bboxes.json",
        help="Output JSON file (default: exported_bboxes.json)"
    )
    
    args = parser.parse_args()
    
    # Get number of boxes from user if not specified
    if args.num_boxes is None:
        while True:
            try:
                args.num_boxes = int(input("\n🎯 How many bounding boxes do you want to create? "))
                if args.num_boxes <= 0:
                    print("❌ Please enter a positive number")
                    continue
                break
            except ValueError:
                print("❌ Please enter a valid number")
    
    # Create bounding boxes
    bboxes = create_bounding_boxes(
        image_path=args.image_path,
        num_boxes=args.num_boxes,
        verbose=True
    )
    
    # Save to JSON file
    output_file = Path(args.output)
    with open(output_file, "w") as f:
        json.dump(bboxes, f, indent=2)
    print(f"✓ Saved to: {output_file}\n")
