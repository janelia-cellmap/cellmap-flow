# Bounding Box Generation Implementation

## Overview
Added interactive bounding box generation to the Pipeline Builder via Neuroglancer integration.

## Features

### 1. **Input Node UI Enhancement**
- Added "📦 Generate" button to INPUT nodes
- Displays count of created bounding boxes
- Click to launch the BBX generation workflow

### 2. **Workflow**
The bounding box generation follows this sequence:

1. **User clicks "Generate" button on INPUT node**
   - Modal appears asking for number of bounding boxes to create

2. **User enters number of boxes**
   - Valid range: 1-100 boxes

3. **Neuroglancer viewer launches in modal**
   - Shows the dataset image layer
   - Has annotation layer ready for drawing
   - Instructions displayed for drawing boxes

4. **User draws bounding boxes**
   - Hold CTRL + Click to set first corner
   - Hold CTRL + Click again to set opposite corner
   - Repeat for each box
   - Real-time count displayed

5. **User clicks "Done - Save Boxes"**
   - Bounding boxes are extracted from viewer
   - Stored in input node parameters
   - Modal closes
   - UI updates with box count

### 3. **YAML Export**
Bounding boxes are automatically included in YAML export:

```yaml
inputs:
  - id: input-123456789
    dataset_path: /path/to/dataset.zarr
    bounding_boxes:
      - offset: [100, 200, 300]
        shape: [50, 60, 70]
      - offset: [150, 250, 350]
        shape: [60, 70, 80]
    position:
      x: 20
      y: 20
```

## Implementation Details

### Frontend Changes
**File: `cellmap_flow/dashboard/templates/pipeline_builder_v2.html`**

#### Modals Added:
1. **BBX Count Modal** (`#bbx-count-modal`)
   - Input field for number of boxes
   - Start button to launch Neuroglancer

2. **BBX Viewer Modal** (`#bbx-viewer-modal`)
   - Full-screen Neuroglancer iframe
   - Status display (boxes created/target)
   - Done button to finalize

#### Functions Added:
- `openBBXGeneratorModal(inputNodeId)` - Open count input modal
- `startBBXGeneration()` - Initialize Neuroglancer and launch viewer
- `pollBBXGeneration()` - Poll for status updates (2s interval)
- `closeBBXViewerModal()` - Close the viewer modal
- `finalizeBBXGeneration()` - Save boxes to input node

#### UI Updates:
- Input node rendering now includes BBX button and display
- YAML export includes bounding_boxes array for input nodes

### Backend Changes
**File: `cellmap_flow/dashboard/app.py`**

#### Global State:
```python
bbx_generator_state = {
    "dataset_path": None,
    "num_boxes": 0,
    "bounding_boxes": [],
    "viewer_process": None,
    "viewer_url": None
}
```

#### API Endpoints:

1. **POST `/api/bbx-generator`**
   - Starts the Neuroglancer viewer
   - Returns viewer URL
   - Input: dataset_path, num_boxes
   - Output: viewer_url, success status

2. **GET `/api/bbx-generator/status`**
   - Returns current status
   - Output: dataset_path, num_boxes, bounding_boxes, count

3. **POST `/api/bbx-generator/finalize`**
   - Finalizes generation and returns boxes
   - Clears state for next generation
   - Output: bounding_boxes, count, success status

### Helper Module
**File: `cellmap_flow/dashboard/bbx_generator.py`**

`BBXGenerator` class for managing:
- Neuroglancer viewer initialization
- Image layer setup
- Annotation layer setup
- Waiting for user input
- Extracting bounding boxes from annotations
- Status polling

Key methods:
- `start_viewer()` - Launch Neuroglancer with proper config
- `wait_for_boxes()` - Poll viewer for annotations
- `_extract_boxes()` - Extract bboxes from viewer state
- `close()` - Clean up resources

## Data Flow

```
User clicks Generate
    ↓
Prompt for number of boxes
    ↓
POST /api/bbx-generator (start viewer)
    ↓
Neuroglancer modal opens
    ↓
GET /api/bbx-generator/status (polling every 2s)
    ↓
User draws boxes in Neuroglancer
    ↓
User clicks "Done - Save Boxes"
    ↓
POST /api/bbx-generator/finalize (extract + return boxes)
    ↓
Update INPUT node with bounding_boxes array
    ↓
Render updated UI with box count
```

## Integration with Pipeline

The bounding boxes are:
1. **Stored in input node** as `params.bounding_boxes`
2. **Exported to YAML** with offset and shape
3. **Available for pipeline execution** - can be used to limit processing region

## Future Enhancements

1. **Viewer Integration**
   - Directly embed Neuroglancer without iframe
   - Real-time annotation synchronization

2. **Box Naming**
   - Allow users to name individual boxes (roi_1, roi_2, etc.)
   - Store descriptive metadata

3. **Box Validation**
   - Validate box coordinates
   - Check for overlaps
   - Warn about invalid geometries

4. **Import/Export**
   - Save/load box configurations to JSON
   - Reuse boxes across projects

5. **Visualization**
   - Show box count and thumbnails in input node
   - Preview boxes on the canvas

## Testing

To test the implementation:

1. **Start the dashboard:**
   ```bash
   python -m cellmap_flow.dashboard
   ```

2. **Navigate to pipeline builder:**
   - Click "Pipeline Builder" in dashboard

3. **Add INPUT node:**
   - Drag INPUT from sidebar
   - Set dataset_path

4. **Generate bounding boxes:**
   - Click "📦 Generate" button
   - Enter number of boxes (e.g., 3)
   - Click "Start Drawing"

5. **Draw boxes in Neuroglancer:**
   - Follow on-screen instructions
   - Create the specified number of boxes

6. **Save boxes:**
   - Click "Done - Save Boxes"
   - Verify count updates in input node

7. **Export YAML:**
   - Click "📥 Export YAML"
   - Check that bounding_boxes array is present

## Notes

- Bounding boxes are stored in the input node state
- They persist through YAML export/import
- The viewer runs on localhost with a free port
- Polling interval is 2 seconds (adjustable)
- Each box stores `offset` (min corner) and `shape` (dimensions)
- Coordinates are in [z, y, x] order
