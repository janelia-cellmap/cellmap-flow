import React, { useState } from 'react';

export const PipelineExporter = ({
  selectedNode,
  nodes,
  edges,
  onExport,
  onImport,
  onUpdateParams,
  onDeleteNode,
}) => {
  const [paramInputs, setParamInputs] = useState({});

  const selectedNodeData = nodes.find((n) => n.id === selectedNode);

  const handleParamChange = (key, value) => {
    setParamInputs({ ...paramInputs, [key]: value });
  };

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
    const pipelineData = onExport();
    
    // Extract workflow structure
    const workflow = {
      input_normalizers: [],
      postprocessors: [],
    };

    // Order nodes by edges
    nodes.forEach((node) => {
      if (node.type === 'normalizer') {
        workflow.input_normalizers.push({
          name: node.data.name,
          params: node.data.params || {},
        });
      } else if (node.type === 'postprocessor') {
        workflow.postprocessors.push({
          name: node.data.name,
          params: node.data.params || {},
        });
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
          const data = file.name.endsWith('.yaml')
            ? parseYAML(content)
            : JSON.parse(content);
          onImport(data);
        } catch (error) {
          alert('Error importing file: ' + error.message);
        }
      };
      reader.readAsText(file);
    }
  };

  return (
    <div style={{
      width: '300px',
      padding: '15px',
      background: '#fff',
      borderLeft: '1px solid #ddd',
      overflowY: 'auto',
      display: 'flex',
      flexDirection: 'column',
    }}>
      <h3>Pipeline Controls</h3>

      {/* Node Inspector */}
      {selectedNodeData && selectedNodeData.type !== 'input' && selectedNodeData.type !== 'output' && (
        <div style={{ marginBottom: '20px', padding: '10px', background: '#f9f9f9', borderRadius: '5px' }}>
          <h4 style={{ marginTop: 0 }}>{selectedNodeData.data.label}</h4>
          <p style={{ fontSize: '12px', color: '#666' }}>Node ID: {selectedNode}</p>

          <label style={{ fontWeight: 'bold', fontSize: '12px' }}>Parameters:</label>
          <div style={{ marginTop: '8px', marginBottom: '10px' }}>
            <input
              type="text"
              placeholder='JSON params e.g. {"param1": 0.5}'
              onChange={(e) => setParamInputs({ param_input: e.target.value })}
              style={{ width: '100%', padding: '5px', fontSize: '11px' }}
            />
          </div>

          <button
            onClick={applyParams}
            style={{
              width: '100%',
              padding: '6px',
              background: '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '12px',
              marginBottom: '8px',
            }}
          >
            Apply Parameters
          </button>

          <button
            onClick={() => onDeleteNode(selectedNode)}
            style={{
              width: '100%',
              padding: '6px',
              background: '#f44336',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '12px',
            }}
          >
            Delete Node
          </button>
        </div>
      )}

      {/* Export/Import */}
      <div style={{ marginTop: 'auto' }}>
        <h4>Export Pipeline</h4>
        <button
          onClick={exportToYAML}
          style={{
            width: '100%',
            padding: '8px',
            background: '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            marginBottom: '8px',
          }}
        >
          Export as YAML
        </button>
        <button
          onClick={exportToJSON}
          style={{
            width: '100%',
            padding: '8px',
            background: '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            marginBottom: '8px',
          }}
        >
          Export as JSON
        </button>

        <h4 style={{ marginTop: '15px' }}>Import Pipeline</h4>
        <input
          type="file"
          accept=".yaml,.json"
          onChange={importFromFile}
          style={{ width: '100%', fontSize: '12px' }}
        />
      </div>
    </div>
  );
};

// Helper functions
const generateYAML = (data) => {
  let yaml = '';
  
  yaml += 'input_normalizers:\n';
  data.input_normalizers.forEach((norm) => {
    yaml += `  - name: ${norm.name}\n`;
    yaml += '    params:\n';
    Object.keys(norm.params).forEach((key) => {
      yaml += `      ${key}: ${JSON.stringify(norm.params[key])}\n`;
    });
  });

  yaml += 'postprocessors:\n';
  data.postprocessors.forEach((post) => {
    yaml += `  - name: ${post.name}\n`;
    yaml += '    params:\n';
    Object.keys(post.params).forEach((key) => {
      yaml += `      ${key}: ${JSON.stringify(post.params[key])}\n`;
    });
  });

  return yaml;
};

const parseYAML = (yaml) => {
  // Simple YAML parser for our use case
  const lines = yaml.split('\n');
  const result = {
    input_normalizers: [],
    postprocessors: [],
  };

  let currentSection = null;
  let currentItem = null;

  lines.forEach((line) => {
    const trimmed = line.trim();
    if (trimmed.startsWith('input_normalizers:')) currentSection = 'input_normalizers';
    else if (trimmed.startsWith('postprocessors:')) currentSection = 'postprocessors';
    else if (trimmed.startsWith('- name:')) {
      currentItem = { name: trimmed.replace('- name: ', ''), params: {} };
    } else if (trimmed.startsWith('name:') && !trimmed.startsWith('- name:')) {
      if (currentItem) currentItem.name = trimmed.replace('name: ', '');
    } else if (trimmed.startsWith('params:')) {
      // params section
    } else if (trimmed && currentItem && !trimmed.startsWith('-')) {
      const [key, value] = trimmed.split(':').map((s) => s.trim());
      if (key && value) {
        try {
          currentItem.params[key] = JSON.parse(value);
        } catch {
          currentItem.params[key] = value;
        }
      }
    }

    // Save item when moving to next
    if ((trimmed.startsWith('- name:') || trimmed.startsWith('name:')) && currentItem && Object.keys(currentItem.params).length > 0) {
      if (currentSection) result[currentSection].push(currentItem);
      currentItem = null;
    }
  });

  // Push last item
  if (currentItem && currentSection) result[currentSection].push(currentItem);

  return result;
};

const downloadFile = (content, filename, mimeType) => {
  const element = document.createElement('a');
  element.setAttribute('href', 'data:' + mimeType + ';charset=utf-8,' + encodeURIComponent(content));
  element.setAttribute('download', filename);
  element.style.display = 'none';
  document.body.appendChild(element);
  element.click();
  document.body.removeChild(element);
};
