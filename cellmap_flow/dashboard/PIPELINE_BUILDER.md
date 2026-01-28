# CellMapFlow Pipeline Builder

A drag-and-drop visual pipeline builder for creating and managing CellMapFlow inference workflows.

## Features

- **Visual Pipeline Editor**: Drag-and-drop interface to connect normalization models and postprocessors
- **Node-based Workflow**: Connect input → normalizers → postprocessors → output
- **Parameter Configuration**: Set parameters for each processing step
- **Export/Import**: Save pipelines as YAML or JSON and reload them later
- **Real-time Validation**: Validate pipelines before applying them

## Architecture

### Frontend (React Flow)

The pipeline builder uses **React Flow** for the visual node editor:

**Components:**
- [pipeline-builder.js](static/js/pipeline-builder.js) - Main React component
- [pipeline-nodes.js](static/js/pipeline-nodes.js) - Node styling and layout
- [pipeline-toolbar.js](static/js/pipeline-toolbar.js) - Toolbar for adding nodes
- [pipeline-exporter.js](static/js/pipeline-exporter.js) - Export/Import functionality
- [pipeline_builder.html](templates/pipeline_builder.html) - HTML template

### Backend (Flask)

Flask routes handle validation and application of pipelines:

**Endpoints:**
- `GET /pipeline-builder` - Render the pipeline builder UI
- `POST /api/pipeline/validate` - Validate pipeline configuration
- `POST /api/pipeline/apply` - Apply pipeline to inference

## File Structure

```
cellmap_flow/dashboard/
├── app.py                          # Flask app with pipeline routes
├── package.json                    # npm dependencies
├── static/
│   └── js/
│       ├── pipeline-builder.js     # Main React component
│       ├── pipeline-nodes.js       # Node components
│       ├── pipeline-toolbar.js     # Toolbar component
│       ├── pipeline-exporter.js    # Export/import logic
│       └── index-pipeline.js       # Entry point
└── templates/
    └── pipeline_builder.html       # HTML template
```

## Usage

### Access the Pipeline Builder

Navigate to: `http://localhost:PORT/pipeline-builder`

### Creating a Pipeline

1. **Add Normalizers**: Select from dropdown in toolbar to add normalization steps
2. **Add Postprocessors**: Select from dropdown to add post-processing steps
3. **Connect Nodes**: Drag from output handle to input handle to connect steps
4. **Configure Parameters**: Click a node and enter JSON parameters in the panel
5. **Apply**: Use the "Apply Pipeline" button to use the configuration

### Exporting a Pipeline

**As YAML:**
```yaml
input_normalizers:
  - name: StandardNormalizer
    params:
      mean: 0.5
      std: 0.1
postprocessors:
  - name: InstanceSegmentation
    params:
      threshold: 0.5
```

**As JSON:**
```json
{
  "nodes": [...],
  "edges": [...],
  "timestamp": "2024-01-28T..."
}
```

### Importing a Pipeline

Click "Import Pipeline" and select a previously saved `.yaml` or `.json` file.

## Configuration

### Node Types

- **Input** (green): Data entry point
- **Normalizer** (blue): Input normalization steps
- **Postprocessor** (pink): Post-processing operations
- **Output** (gold): Final output

### Parameter Format

Parameters are specified as JSON. Examples:

```json
{
  "clip_min": -1.0,
  "clip_max": 1.0,
  "bias": 1.0,
  "multiplier": 127.5
}
```

## Integration with CellMapFlow

The pipeline builder integrates with your existing CellMapFlow infrastructure:

1. **Normalizers**: Uses `get_input_normalizers()` from [input_normalize.py](../norm/input_normalize.py)
2. **Postprocessors**: Uses `get_postprocessors_list()` from [postprocessors.py](../post/postprocessors.py)
3. **Validation**: Validates against available models before applying
4. **Application**: Applies to global state (`g.input_norms`, `g.postprocess`)

## API Reference

### GET /pipeline-builder

Returns the pipeline builder UI with available normalizers and postprocessors.

**Response:**
- HTML page with React Flow editor

### POST /api/pipeline/validate

Validates a pipeline configuration.

**Request:**
```json
{
  "input_normalizers": [
    { "name": "StandardNormalizer", "params": {...} }
  ],
  "postprocessors": [
    { "name": "InstanceSegmentation", "params": {...} }
  ]
}
```

**Response:**
```json
{
  "valid": true,
  "message": "Pipeline is valid"
}
```

### POST /api/pipeline/apply

Applies a validated pipeline to the current inference.

**Request:** Same as validate

**Response:**
```json
{
  "message": "Pipeline applied successfully",
  "normalizers_applied": 2,
  "postprocessors_applied": 1
}
```

## Development

### Setup

```bash
cd cellmap_flow/dashboard
npm install
npm run build  # Build for production
npm run dev    # Watch mode for development
```

### Building the Frontend

React Flow components are bundled using webpack. Run `npm run build` to generate the bundled JavaScript.

## Dependencies

### Frontend
- **React** 18.2+ - UI library
- **React DOM** 18.2+ - DOM rendering
- **React Flow** 11.10+ - Node-based visual editor

### Backend
- **Flask** - Web framework (already in CellMapFlow)
- **Pydantic** - Configuration validation (already in CellMapFlow)

## Future Enhancements

- [ ] Real-time pipeline preview/simulation
- [ ] Custom node templates for complex operations
- [ ] Pipeline library/templates for common workflows
- [ ] Performance profiling for pipeline execution
- [ ] Undo/Redo functionality
- [ ] Keyboard shortcuts for faster node creation
- [ ] Search/filter for large model lists

## Troubleshooting

### Pipeline not applying
- Check browser console for errors
- Verify all node names match available normalizers/postprocessors
- Ensure JSON parameter format is valid

### Import fails
- Ensure file is valid YAML or JSON
- Check that all referenced models exist in your installation

### Styling issues
- Clear browser cache
- Rebuild with `npm run build`
