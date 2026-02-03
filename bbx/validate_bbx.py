#%%

import neuroglancer
import json
from pathlib import Path

# Configure Neuroglancer server

neuroglancer.set_server_bind_address("0.0.0.0")

# Image layer path
image_path = "https://cellmap-vm1.int.janelia.org/nrs/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8/"

# Load bounding boxes from JSON file
json_file = Path("exported_bboxes.json")

if not json_file.exists():
    print(f"❌ Error: {json_file} not found!")
    print("Run 'python create_and_export_bboxes.py' first to create bounding boxes.")
    exit(1)

with open(json_file, "r") as f:
    bounding_boxes = json.load(f)

print(f"\n✅ Loaded {len(bounding_boxes)} bounding box(es) from {json_file}\n")

# Create viewer
viewer = neuroglancer.Viewer()

# Add layers with proper coordinate space
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
    
    # Add LocalAnnotationLayer for bounding boxes
    s.layers["bboxes"] = neuroglancer.LocalAnnotationLayer(
        dimensions=neuroglancer.CoordinateSpace(
            names=["z", "y", "x"],
            units="nm",
            scales=[1, 1, 1],
        ),
    )
#     )
    
    # Add bounding boxes as AxisAlignedBoundingBoxAnnotation
    for bbox in bounding_boxes:
        offset = bbox["offset"]
        shape = bbox["shape"]
        
        point_a = [float(offset[0]), float(offset[1]), float(offset[2])]
        point_b = [
            float(offset[0] + shape[0]),
            float(offset[1] + shape[1]),
            float(offset[2] + shape[2])
        ]
        
        s.layers["bboxes"].annotations.append(
            neuroglancer.AxisAlignedBoundingBoxAnnotation(
                point_a=point_a,
                point_b=point_b,
                id=bbox["name"],
                description=bbox["description"],
            )
        )
    
    # Set initial view position (center of first bounding box)
    if bounding_boxes:
        first_bbox = bounding_boxes[0]
        center = [
            first_bbox["offset"][i] + first_bbox["shape"][i] // 2 
            for i in range(3)
        ]
        s.position = center

# Print viewer URL
viewer_url = str(viewer)
print("\n" + "=" * 60)
print("Neuroglancer Viewer Ready!")
print("=" * 60)
print(f"URL: {viewer_url}")
print("=" * 60 + "\n")

# Open in browser
try:
    webbrowser.open(viewer_url)
    print("Opening in browser...")
except Exception as e:
    print(f"Could not open browser automatically: {e}")
    print(f"Please open the URL manually: {viewer_url}")

# Keep viewer running
try:
    input("Press Enter to close the viewer...\n")
except KeyboardInterrupt:
    print("\nViewer closed.")

# %%
