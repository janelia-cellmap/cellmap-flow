import React from 'react';
import { Handle, Position } from 'reactflow';

const nodeColors = {
  input: '#90EE90',
  normalizer: '#87CEEB',
  postprocessor: '#FFB6C1',
  output: '#FFD700',
};

export const PipelineNode = ({ data, selected, id }) => {
  const bgColor = nodeColors[data.type] || '#fff';

  return (
    <div
      style={{
        padding: '10px',
        border: `2px ${selected ? 'blue' : '#ddd'} solid`,
        borderRadius: '8px',
        background: bgColor,
        minWidth: '120px',
        textAlign: 'center',
        fontWeight: 'bold',
        cursor: 'pointer',
      }}
    >
      {data.type !== 'input' && (
        <Handle type="target" position={Position.Top} />
      )}
      <div>{data.label}</div>
      {data.type !== 'output' && (
        <Handle type="source" position={Position.Bottom} />
      )}
    </div>
  );
};
