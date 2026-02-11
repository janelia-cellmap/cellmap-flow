"""
Bounding box generator integration for the pipeline builder dashboard.
Uses the neuroglancer viewer to interactively create bounding boxes.
"""

import json
import time
import logging
import neuroglancer
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class BBXGenerator:
    """Manages interactive bounding box generation in Neuroglancer"""
    
    def __init__(self, image_path: str, num_boxes: int, verbose: bool = True):
        """
        Initialize the BBX generator.
        
        Args:
            image_path: URL or path to image (zarr/n5/precomputed)
            num_boxes: Target number of bounding boxes
            verbose: Print progress messages
        """
        self.image_path = image_path
        self.num_boxes = num_boxes
        self.verbose = verbose
        self.viewer = None
        self.bounding_boxes = []
        
    def start_viewer(self) -> str:
        """
        Start Neuroglancer viewer for drawing bounding boxes.
        
        Returns:
            URL to access the viewer
        """
        try:
            neuroglancer.set_server_bind_address("0.0.0.0")
            self.viewer = neuroglancer.Viewer()
            
            with self.viewer.txn() as s:
                # Set coordinate space
                s.dimensions = neuroglancer.CoordinateSpace(
                    names=["z", "y", "x"],
                    units="nm",
                    scales=[8, 8, 8],
                )
                
                # Add image layer
                s.layers["fibsem"] = neuroglancer.ImageLayer(
                    source=f"zarr2://{self.image_path}",
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
            
            viewer_url = str(self.viewer)
            
            if self.verbose:
                print("=" * 70)
                print("Neuroglancer Bounding Box Creator (Ctrl+Click to Draw)")
                print("=" * 70)
                print(f"URL: {viewer_url}")
                print("\nHOW TO CREATE BOUNDING BOXES:")
                print("  1. Open the Neuroglancer URL above")
                print("  2. Click on 'annotations' layer in LEFT PANEL")
                print("  3. Click the BOUNDING BOX TOOL icon (□ symbol)")
                print("  4. Hold CTRL and CLICK to set first corner")
                print("  5. Hold CTRL and CLICK again to set opposite corner")
                print(f"\n🎯 TARGET: {self.num_boxes} box(es)")
                print("⏱️  Will auto-close when done\n")
                print("=" * 70 + "\n")
            
            return viewer_url
            
        except Exception as e:
            logger.error(f"Error starting viewer: {str(e)}")
            raise
    
    def wait_for_boxes(self, timeout: int = 3600) -> List[Dict[str, Any]]:
        """
        Wait for user to create bounding boxes.
        Polls the viewer annotations and returns when target is reached or timeout.
        
        Args:
            timeout: Maximum seconds to wait (default 1 hour)
            
        Returns:
            List of bounding boxes
        """
        start_time = time.time()
        last_count = 0
        check_interval = 1  # Check every 1 second
        
        while time.time() - start_time < timeout:
            try:
                with self.viewer.txn() as s:
                    try:
                        annotations_layer = s.layers["annotations"]
                        if annotations_layer:
                            current_count = len(annotations_layer.annotations)
                            
                            # Print status updates
                            if current_count != last_count:
                                if self.verbose:
                                    print(f"📦 Boxes created: {current_count}/{self.num_boxes}")
                                last_count = current_count
                                
                                # Check if target reached
                                if current_count >= self.num_boxes:
                                    if self.verbose:
                                        print(f"\n✅ Target reached! Extracting {self.num_boxes} bounding box(es)...\n")
                                    break
                    except (KeyError, IndexError):
                        # Layer doesn't exist yet or no annotations
                        pass
                
                time.sleep(check_interval)
            
            except KeyboardInterrupt:
                if self.verbose:
                    print("\n\n⚠️  Cancelled by user. Extracting current boxes...\n")
                break
        
        # Extract the bounding boxes
        self.bounding_boxes = self._extract_boxes()
        return self.bounding_boxes
    
    def _extract_boxes(self) -> List[Dict[str, Any]]:
        """Extract bounding boxes from viewer annotations"""
        try:
            with self.viewer.txn() as s:
                viewer_state = s.to_json()
            
            bboxes_list = []
            
            for layer in viewer_state.get("layers", []):
                if layer.get("name") == "annotations":
                    annotations = layer.get("annotations", [])
                    
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
            
            if self.verbose:
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
        
        except Exception as e:
            logger.error(f"Error extracting boxes: {str(e)}")
            return []
    
    def close(self):
        """Close the viewer"""
        if self.viewer:
            self.viewer = None


def create_bounding_boxes_interactive(
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
        List of bounding boxes
    """
    generator = BBXGenerator(image_path, num_boxes, verbose)
    
    try:
        # Start the viewer
        url = generator.start_viewer()
        
        # Wait for boxes to be created
        bboxes = generator.wait_for_boxes()
        
        return bboxes
    
    finally:
        generator.close()
