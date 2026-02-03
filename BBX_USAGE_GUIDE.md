# Bounding Box Generation - Usage Guide

## Quick Start

### Step 1: Open Pipeline Builder
1. Start the cellmap-flow dashboard:
   ```bash
   python -m cellmap_flow.dashboard
   ```
2. Navigate to "Pipeline Builder" section

### Step 2: Create INPUT Node
1. Drag "🟣 INPUT" from the "Flow Control" section onto the canvas
2. The INPUT node appears with a dataset path field

### Step 3: Set Dataset Path
1. In the INPUT node, set the "Dataset Path" to your image location
   - Example: `s3://bucket/dataset.zarr`
   - Or: `/path/to/local/dataset.zarr`
2. Click 💾 Save

### Step 4: Generate Bounding Boxes
1. Click the "📦 Generate" button in the INPUT node
2. A modal appears asking "How many bounding boxes?"
3. Enter the number (1-100)
4. Click "✓ Start Drawing"

### Step 5: Draw Boxes in Neuroglancer
The Neuroglancer viewer loads with your image. To draw a bounding box:

1. Look for the annotation layer (should be visible in left panel)
2. Select the Bounding Box tool (□ icon)
3. **First corner:** Hold Ctrl + Click
4. **Opposite corner:** Hold Ctrl + Click again
5. Repeat for each box

The count updates in real-time at the bottom of the modal.

### Step 6: Save Boxes
Once you've drawn all boxes:
1. Click "✓ Done - Save Boxes"
2. The modal closes
3. The INPUT node now shows "📦 Generate (3 bbox(es))" (or your count)

### Step 7: Export Pipeline
1. Click "📥 Export YAML" to download your pipeline
2. Your YAML now includes the bounding boxes:

```yaml
inputs:
  - id: input-12345
    dataset_path: /path/to/dataset.zarr
    bounding_boxes:
      - offset: [100, 200, 300]
        shape: [50, 60, 70]
      - offset: [150, 250, 350]
        shape: [60, 70, 80]
      - offset: [200, 300, 400]
        shape: [55, 65, 75]
    position:
      x: 20
      y: 20
```

## UI Components

### INPUT Node with Bounding Boxes

```
┌─────────────────────────────────┐
│ INPUT              💾  🗑️       │
├─────────────────────────────────┤
│ Dataset Path:                   │
│ [/path/to/dataset.zarr        ] │
│                                 │
│ Bounding Boxes                  │
│ [📦 Generate (3 bbox(es))    ]  │
│ 3 bbox(es)                      │
└─────────────────────────────────┘
```

### BBX Count Modal

```
┌─────────────────────────────────┐
│ Generate Bounding Boxes       ✕ │
├─────────────────────────────────┤
│ How many bounding boxes do you  │
│ want to create?                 │
│                                 │
│ [         1            ]         │
│                                 │
│ [Cancel] [✓ Start Drawing]     │
└─────────────────────────────────┘
```

### BBX Viewer Modal

```
┌──────────────────────────────────────┐
│ Draw Bounding Boxes in Neuroglancer ✕ │
├──────────────────────────────────────┤
│                                      │
│  ┌──────────────────────────────┐   │
│  │                              │   │
│  │   [Neuroglancer Viewer]      │   │
│  │                              │   │
│  │   (Shows image with          │   │
│  │    annotation layer)         │   │
│  │                              │   │
│  └──────────────────────────────┘   │
│                                      │
│ 📦 2/3 box(es) created              │
├──────────────────────────────────────┤
│ [Cancel] [✓ Done - Save Boxes]     │
└──────────────────────────────────────┘
```

## Keyboard Shortcuts (Neuroglancer)

| Action | Key |
|--------|-----|
| Set first corner | **Ctrl + Click** |
| Set opposite corner | **Ctrl + Click** |
| Pan view | Middle mouse / Space + Drag |
| Zoom in/out | Scroll wheel |
| Reset view | Home key |

## Data Format

### Bounding Box Structure

Each bounding box in the YAML/JSON has:

```json
{
  "offset": [z, y, x],    // Minimum corner coordinates
  "shape": [dz, dy, dx]   // Size in each dimension
}
```

**Coordinate System:**
- Origin (0,0,0) is top-left-front
- Z increases downward
- Y increases downward
- X increases rightward
- Units are in voxels (or whatever the dataset uses)

### Example Bounding Box

```python
{
  "offset": [100, 200, 300],   # Start at z=100, y=200, x=300
  "shape": [50, 60, 70]         # Size: 50 deep, 60 tall, 70 wide
}

# This box covers:
# z: 100-149 (50 slices)
# y: 200-259 (60 pixels)
# x: 300-369 (70 pixels)
```

## Troubleshooting

### "Please set the dataset path on the INPUT node first"
**Problem:** Clicked Generate without setting dataset path
**Solution:** 
1. Enter dataset path in the "Dataset Path" field
2. Click 💾 Save
3. Try Generate again

### Neuroglancer doesn't load
**Problem:** Viewer modal is blank
**Solution:**
1. Check browser console for errors (F12)
2. Ensure Neuroglancer server is running
3. Check dataset path is accessible
4. Try refreshing the page

### Boxes not appearing
**Problem:** Drew boxes but they're not visible
**Solution:**
1. Ensure annotation layer is selected in left panel
2. Zoom in/out to make sure boxes are in view
3. Check that bounding box tool (□) is selected

### Can't draw boxes
**Problem:** Ctrl+Click doesn't create corners
**Solution:**
1. Make sure to click ON the image area
2. Check that annotation layer is active
3. Verify bounding box tool is selected
4. Try a different browser if issue persists

### Boxes saved but don't appear in export
**Problem:** YAML doesn't include bounding_boxes
**Solution:**
1. Make sure you clicked "Done - Save Boxes"
2. Check that the count updated in the INPUT node
3. Re-generate if needed
4. Try exporting again

## Advanced Usage

### Importing Bounding Boxes

If you have an existing `pipeline.yaml` with bounding boxes:

1. Click "📂 Import YAML"
2. Select your YAML file
3. The bounding boxes will be loaded
4. Click the "📦 Generate" button to view/edit them

### Editing Existing Boxes

To modify boxes:

1. Click "📦 Generate" on the INPUT node
2. You'll get a fresh viewer (previous boxes not shown)
3. Draw the new set of boxes
4. Click "Done - Save Boxes" to replace

### Multiple Regions

Create multiple INPUT nodes to define different regions:

```
INPUT (Region A)
├─ dataset_path: /data/region-a.zarr
└─ bounding_boxes: [box1, box2]

INPUT (Region B)
├─ dataset_path: /data/region-b.zarr
└─ bounding_boxes: [box3, box4]
```

Each region has its own bounding boxes stored separately.

## API Reference (For Developers)

### Starting BBX Generation

```javascript
// Frontend call
const response = await fetch('/api/bbx-generator', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        dataset_path: "/path/to/dataset.zarr",
        num_boxes: 3
    })
});
const result = await response.json();
// result.viewer_url contains the Neuroglancer URL
```

### Getting Status

```javascript
const response = await fetch('/api/bbx-generator/status');
const result = await response.json();
// result.bounding_boxes contains current boxes
// result.count is the number of boxes
```

### Finalizing Generation

```javascript
const response = await fetch('/api/bbx-generator/finalize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({})
});
const result = await response.json();
// result.bounding_boxes contains final boxes
```

## Performance Notes

- Polling interval: 2 seconds (adjustable)
- Viewer loads in iframe for isolation
- Boxes stored in memory during session
- No database persistence (stored in YAML export)

## Known Limitations

1. **No undo/redo** - Redraw boxes if needed
2. **No box editing** - Must redraw entire set
3. **No automatic validation** - User responsible for correct geometry
4. **Single dataset** - One dataset per INPUT node
5. **Linear YAML** - Boxes stored in simple offset/shape format

## Future Enhancements

- [ ] Named bounding boxes (roi_1, roi_2, etc.)
- [ ] Visual preview of boxes on canvas
- [ ] Box validation and overlap checking
- [ ] Import from JSON/CSV files
- [ ] Batch create boxes via coordinates
- [ ] Copy/paste boxes between nodes
- [ ] 3D visualization of boxes
