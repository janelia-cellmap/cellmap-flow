// In-browser bundle for the pipeline builder — uses UMD React and ReactFlow globals
// This file is transpiled by Babel in the browser (development convenience).

// Defer accessing ReactFlow UMD globals until runtime to avoid errors
let ReactFlowProvider, ReactFlow, addEdge, useNodesState, useEdgesState, Controls, Background, useReactFlow, Handle, Position;
const { useCallback, useState } = React;

const nodeTypes = {
  normalizer: PipelineNode,
  postprocessor: PipelineNode,
  input: PipelineNode,
  output: PipelineNode,
};

function PipelineNode({ data, selected, id }) {
  const nodeColors = {
    input: '#90EE90',
    normalizer: '#87CEEB',
    postprocessor: '#FFB6C1',
    output: '#FFD700',
  };
  const bgColor = nodeColors[data.type] || '#fff';

  return (
    React.createElement('div', {
      style: {
        padding: '10px',
        border: `2px ${selected ? 'blue' : '#ddd'} solid`,
        borderRadius: '8px',
        background: bgColor,
        minWidth: '120px',
        textAlign: 'center',
        fontWeight: 'bold',
        cursor: 'pointer',
      },
    },
      data.type !== 'input' && React.createElement(Handle, { type: 'target', position: Position.Top }),
      React.createElement('div', null, data.label),
      data.type !== 'output' && React.createElement(Handle, { type: 'source', position: Position.Bottom })
    )
  );
}

function PipelineToolbar({
  availableNormalizers,
  availablePostprocessors,
  onAddNormalizer,
  onAddPostprocessor,
}) {
  return (
    React.createElement('div', { style: {
      padding: '10px',
      background: '#f5f5f5',
      borderBottom: '1px solid #ddd',
      display: 'flex',
      gap: '10px',
      alignItems: 'center',
    } },
      React.createElement('label', { style: { fontWeight: 'bold' } }, 'Add Normalizer:'),
      React.createElement('select', {
        onChange: (e) => {
          if (e.target.value) {
            onAddNormalizer(e.target.value);
            e.target.value = '';
          }
        }
      },
        React.createElement('option', { value: '' }, 'Select a normalizer...'),
        availableNormalizers && Object.keys(availableNormalizers).map((norm) => React.createElement('option', { key: norm, value: norm }, norm))
      ),

      React.createElement('label', { style: { fontWeight: 'bold', marginLeft: '20px' } }, 'Add Postprocessor:'),
      React.createElement('select', {
        onChange: (e) => {
          if (e.target.value) {
            onAddPostprocessor(e.target.value);
            e.target.value = '';
          }
        }
      },
        React.createElement('option', { value: '' }, 'Select a postprocessor...'),
        availablePostprocessors && Object.keys(availablePostprocessors).map((post) => React.createElement('option', { key: post, value: post }, post))
      )
    )
  );
}

function PipelineExporter({
  selectedNode,
  nodes,
  edges,
  onExport,
  onImport,
  onUpdateParams,
  onDeleteNode,
}) {
  const [paramInputs, setParamInputs] = useState({});
  const selectedNodeData = nodes.find((n) => n.id === selectedNode);

  const applyParams = () => {
    if (selectedNode && selectedNodeData) {
      try {
        const params = {};
        Object.keys(paramInputs).forEach((key) => {
          try {
            params[key] = JSON.parse(paramInputs[key]);
          } catch {
            params[key] = paramInputs[key];
          }
        });
        onUpdateParams(selectedNode, params);
        setParamInputs({});
      } catch (error) {
        alert('Invalid parameter format: ' + error.message);
      }
    }
  };

  const exportToYAML = () => {
    const workflow = {
      input_normalizers: [],
      postprocessors: [],
    };
    nodes.forEach((node) => {
      if (node.type === 'normalizer') {
        workflow.input_normalizers.push({ name: node.data.name, params: node.data.params || {} });
      } else if (node.type === 'postprocessor') {
        workflow.postprocessors.push({ name: node.data.name, params: node.data.params || {} });
      }
    });
    const yamlContent = generateYAML(workflow);
    downloadFile(yamlContent, 'pipeline.yaml', 'text/yaml');
  };

  const exportToJSON = () => {
    const pipelineData = onExport();
    const jsonContent = JSON.stringify(pipelineData, null, 2);
    downloadFile(jsonContent, 'pipeline.json', 'application/json');
  };

  const importFromFile = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const content = e.target.result;
          const data = file.name.endsWith('.yaml') ? parseYAML(content) : JSON.parse(content);
          onImport(data);
        } catch (error) {
          alert('Error importing file: ' + error.message);
        }
      };
      reader.readAsText(file);
    }
  };

  return (
    React.createElement('div', { style: {
      width: '300px',
      padding: '15px',
      background: '#fff',
      borderLeft: '1px solid #ddd',
      overflowY: 'auto',
      display: 'flex',
      flexDirection: 'column',
    } },
      React.createElement('h3', null, 'Pipeline Controls'),
      selectedNodeData && selectedNodeData.type !== 'input' && selectedNodeData.type !== 'output' && (
        React.createElement('div', { style: { marginBottom: '20px', padding: '10px', background: '#f9f9f9', borderRadius: '5px' } },
          React.createElement('h4', { style: { marginTop: 0 } }, selectedNodeData.data.label),
          React.createElement('p', { style: { fontSize: '12px', color: '#666' } }, `Node ID: ${selectedNode}`),
          React.createElement('label', { style: { fontWeight: 'bold', fontSize: '12px' } }, 'Parameters:'),
          React.createElement('div', { style: { marginTop: '8px', marginBottom: '10px' } },
            React.createElement('input', {
              type: 'text',
              placeholder: '{"param1": 0.5}',
              onChange: (e) => setParamInputs({ param_input: e.target.value }),
              style: { width: '100%', padding: '5px', fontSize: '11px' },
            })
          ),
          React.createElement('button', { onClick: applyParams, style: { width: '100%', padding: '6px', background: '#4CAF50', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '12px', marginBottom: '8px' } }, 'Apply Parameters'),
          React.createElement('button', { onClick: () => onDeleteNode(selectedNode), style: { width: '100%', padding: '6px', background: '#f44336', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '12px' } }, 'Delete Node')
        )
      ),
      React.createElement('div', { style: { marginTop: 'auto' } },
        React.createElement('h4', null, 'Export Pipeline'),
        React.createElement('button', { onClick: exportToYAML, style: { width: '100%', padding: '8px', background: '#2196F3', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', marginBottom: '8px' } }, 'Export as YAML'),
        React.createElement('button', { onClick: exportToJSON, style: { width: '100%', padding: '8px', background: '#2196F3', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', marginBottom: '8px' } }, 'Export as JSON'),
        React.createElement('h4', { style: { marginTop: '15px' } }, 'Import Pipeline'),
        React.createElement('input', { type: 'file', accept: '.yaml,.json', onChange: importFromFile, style: { width: '100%', fontSize: '12px' } })
      )
    )
  );
}

function PipelineBuilder({ availableNormalizers, availablePostprocessors }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([
    { id: 'input', data: { label: 'Input Data', type: 'input' }, position: { x: 0, y: 0 }, type: 'input' },
    { id: 'output', data: { label: 'Output', type: 'output' }, position: { x: 400, y: 300 }, type: 'output' },
  ]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const reactFlowInstance = useReactFlow();

  const onConnectCb = useCallback((connection) => setEdges((eds) => addEdge(connection, eds)), [setEdges]);

  const addNormalizer = useCallback((normalizerName) => {
    const id = `norm-${Date.now()}`;
    const newNode = { id, type: 'normalizer', data: { label: normalizerName, type: 'normalizer', name: normalizerName, params: {} }, position: { x: 100, y: 150 } };
    setNodes((nds) => [...nds, newNode]);
  }, [setNodes]);

  const addPostprocessor = useCallback((processorName) => {
    const id = `post-${Date.now()}`;
    const newNode = { id, type: 'postprocessor', data: { label: processorName, type: 'postprocessor', name: processorName, params: {} }, position: { x: 250, y: 150 } };
    setNodes((nds) => [...nds, newNode]);
  }, [setNodes]);

  const updateNodeParams = useCallback((nodeId, params) => setNodes((nds) => nds.map((node) => node.id === nodeId ? { ...node, data: { ...node.data, params } } : node)), [setNodes]);

  const deleteNode = useCallback((nodeId) => { setNodes((nds) => nds.filter((node) => node.id !== nodeId)); setEdges((eds) => eds.filter((edge) => edge.source !== nodeId && edge.target !== nodeId)); }, [setNodes, setEdges]);

  const exportPipeline = useCallback(() => ({ nodes, edges, timestamp: new Date().toISOString() }), [nodes, edges]);
  const importPipeline = useCallback((pipelineData) => { if (pipelineData.nodes) setNodes(pipelineData.nodes); if (pipelineData.edges) setEdges(pipelineData.edges); }, [setNodes, setEdges]);

  return (
    React.createElement('div', { style: { width: '100%', height: '100vh', display: 'flex', flexDirection: 'column' } },
      React.createElement(PipelineToolbar, { availableNormalizers, availablePostprocessors, onAddNormalizer: addNormalizer, onAddPostprocessor: addPostprocessor }),
      React.createElement('div', { style: { flex: 1, display: 'flex' } },
        React.createElement('div', { style: { flex: 1 } },
          React.createElement(ReactFlow, { nodes, edges, onNodesChange, onEdgesChange, onConnect: onConnectCb, nodeTypes, onNodeClick: (event, node) => setSelectedNode(node.id) },
            React.createElement(Background, null),
            React.createElement(Controls, null)
          )
        ),
        React.createElement(PipelineExporter, { selectedNode, nodes, edges, onExport: exportPipeline, onImport: importPipeline, onUpdateParams: updateNodeParams, onDeleteNode: deleteNode })
      )
    )
  );
}

// Helpers
function generateYAML(data) {
  let yaml = '';
  yaml += 'input_normalizers:\n';
  data.input_normalizers.forEach((norm) => {
    yaml += `  - name: ${norm.name}\n`;
    yaml += '    params:\n';
    Object.keys(norm.params).forEach((key) => { yaml += `      ${key}: ${JSON.stringify(norm.params[key])}\n`; });
  });
  yaml += 'postprocessors:\n';
  data.postprocessors.forEach((post) => {
    yaml += `  - name: ${post.name}\n`;
    yaml += '    params:\n';
    Object.keys(post.params).forEach((key) => { yaml += `      ${key}: ${JSON.stringify(post.params[key])}\n`; });
  });
  return yaml;
}

function parseYAML(yaml) {
  const lines = yaml.split('\n');
  const result = { input_normalizers: [], postprocessors: [] };
  let currentSection = null;
  let currentItem = null;
  lines.forEach((line) => {
    const trimmed = line.trim();
    if (trimmed.startsWith('input_normalizers:')) currentSection = 'input_normalizers';
    else if (trimmed.startsWith('postprocessors:')) currentSection = 'postprocessors';
    else if (trimmed.startsWith('- name:')) { currentItem = { name: trimmed.replace('- name: ', ''), params: {} }; }
    else if (trimmed.startsWith('name:') && !trimmed.startsWith('- name:')) { if (currentItem) currentItem.name = trimmed.replace('name: ', ''); }
    else if (trimmed.startsWith('params:')) {}
    else if (trimmed && currentItem && !trimmed.startsWith('-')) {
      const [key, value] = trimmed.split(':').map((s) => s.trim());
      if (key && value) {
        try { currentItem.params[key] = JSON.parse(value); } catch { currentItem.params[key] = value; }
      }
    }
    if ((trimmed.startsWith('- name:') || trimmed.startsWith('name:')) && currentItem && Object.keys(currentItem.params).length > 0) { if (currentSection) result[currentSection].push(currentItem); currentItem = null; }
  });
  if (currentItem && currentSection) result[currentSection].push(currentItem);
  return result;
}

function downloadFile(content, filename, mimeType) {
  const element = document.createElement('a');
  element.setAttribute('href', 'data:' + mimeType + ';charset=utf-8,' + encodeURIComponent(content));
  element.setAttribute('download', filename);
  element.style.display = 'none';
  document.body.appendChild(element);
  element.click();
  document.body.removeChild(element);
}

// Mount app
document.addEventListener('DOMContentLoaded', function () {
  const APP_CONFIG = window.APP_CONFIG || { availableNormalizers: {}, availablePostprocessors: {} };

  // Ensure ReactFlow UMD global is available
  const RF = window.ReactFlow;
  if (!RF) {
    console.error(
      'ReactFlow UMD not found on window.ReactFlow. Ensure ReactFlow script is loaded before bundle.'
    );
    return;
  }

  // Extract needed ReactFlow exports at runtime
  ReactFlowProvider = RF.ReactFlowProvider;
  ReactFlow = RF.ReactFlow;
  addEdge = RF.addEdge;
  useNodesState = RF.useNodesState;
  useEdgesState = RF.useEdgesState;
  Controls = RF.Controls;
  Background = RF.Background;
  useReactFlow = RF.useReactFlow;
  Handle = RF.Handle;
  Position = RF.Position;

  ReactDOM.render(
    React.createElement(ReactFlowProvider, null, React.createElement(PipelineBuilder, { availableNormalizers: APP_CONFIG.availableNormalizers, availablePostprocessors: APP_CONFIG.availablePostprocessors })),
    document.getElementById('pipeline-root')
  );
});
