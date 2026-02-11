import React, { useCallback, useState, useRef } from 'react';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  useReactFlow,
  NodeTypes,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { PipelineNode } from './pipeline-nodes';
import { PipelineToolbar } from './pipeline-toolbar';
import { PipelineExporter } from './pipeline-exporter';

const nodeTypes = {
  normalizer: PipelineNode,
  postprocessor: PipelineNode,
  input: PipelineNode,
  output: PipelineNode,
};

export const PipelineBuilder = ({ availableNormalizers, availablePostprocessors }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([
    {
      id: 'input',
      data: { label: 'Input Data', type: 'input' },
      position: { x: 0, y: 0 },
      type: 'input',
    },
    {
      id: 'output',
      data: { label: 'Output', type: 'output' },
      position: { x: 400, y: 300 },
      type: 'output',
    },
  ]);

  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const reactFlowInstance = useReactFlow();

  const onConnect = useCallback(
    (connection) => {
      setEdges((eds) => addEdge(connection, eds));
    },
    [setEdges]
  );

  const addNormalizer = useCallback(
    (normalizerName) => {
      const id = `norm-${Date.now()}`;
      const newNode = {
        id,
        type: 'normalizer',
        data: {
          label: normalizerName,
          type: 'normalizer',
          name: normalizerName,
          params: {},
        },
        position: { x: 100, y: 150 },
      };
      setNodes((nds) => [...nds, newNode]);
    },
    [setNodes]
  );

  const addPostprocessor = useCallback(
    (processorName) => {
      const id = `post-${Date.now()}`;
      const newNode = {
        id,
        type: 'postprocessor',
        data: {
          label: processorName,
          type: 'postprocessor',
          name: processorName,
          params: {},
        },
        position: { x: 250, y: 150 },
      };
      setNodes((nds) => [...nds, newNode]);
    },
    [setNodes]
  );

  const updateNodeParams = useCallback(
    (nodeId, params) => {
      setNodes((nds) =>
        nds.map((node) =>
          node.id === nodeId
            ? { ...node, data: { ...node.data, params } }
            : node
        )
      );
    },
    [setNodes]
  );

  const deleteNode = useCallback(
    (nodeId) => {
      setNodes((nds) => nds.filter((node) => node.id !== nodeId));
      setEdges((eds) =>
        eds.filter((edge) => edge.source !== nodeId && edge.target !== nodeId)
      );
    },
    [setNodes, setEdges]
  );

  const exportPipeline = useCallback(() => {
    return {
      nodes,
      edges,
      timestamp: new Date().toISOString(),
    };
  }, [nodes, edges]);

  const importPipeline = useCallback(
    (pipelineData) => {
      if (pipelineData.nodes) {
        setNodes(pipelineData.nodes);
      }
      if (pipelineData.edges) {
        setEdges(pipelineData.edges);
      }
    },
    [setNodes, setEdges]
  );

  return (
    <div style={{ width: '100%', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <PipelineToolbar
        availableNormalizers={availableNormalizers}
        availablePostprocessors={availablePostprocessors}
        onAddNormalizer={addNormalizer}
        onAddPostprocessor={addPostprocessor}
      />

      <div style={{ flex: 1, display: 'flex' }}>
        <div style={{ flex: 1 }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            onNodeClick={(event, node) => setSelectedNode(node.id)}
          >
            <Background />
            <Controls />
          </ReactFlow>
        </div>

        <PipelineExporter
          selectedNode={selectedNode}
          nodes={nodes}
          edges={edges}
          onExport={exportPipeline}
          onImport={importPipeline}
          onUpdateParams={updateNodeParams}
          onDeleteNode={deleteNode}
        />
      </div>
    </div>
  );
};
