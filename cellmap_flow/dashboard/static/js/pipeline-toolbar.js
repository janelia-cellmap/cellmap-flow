import React from 'react';

export const PipelineToolbar = ({
  availableNormalizers,
  availablePostprocessors,
  onAddNormalizer,
  onAddPostprocessor,
}) => {
  return (
    <div style={{
      padding: '10px',
      background: '#f5f5f5',
      borderBottom: '1px solid #ddd',
      display: 'flex',
      gap: '10px',
      alignItems: 'center',
    }}>
      <label style={{ fontWeight: 'bold' }}>Add Normalizer:</label>
      <select
        onChange={(e) => {
          if (e.target.value) {
            onAddNormalizer(e.target.value);
            e.target.value = '';
          }
        }}
      >
        <option value="">Select a normalizer...</option>
        {availableNormalizers && Object.keys(availableNormalizers).map((norm) => (
          <option key={norm} value={norm}>
            {norm}
          </option>
        ))}
      </select>

      <label style={{ fontWeight: 'bold', marginLeft: '20px' }}>Add Postprocessor:</label>
      <select
        onChange={(e) => {
          if (e.target.value) {
            onAddPostprocessor(e.target.value);
            e.target.value = '';
          }
        }}
      >
        <option value="">Select a postprocessor...</option>
        {availablePostprocessors && Object.keys(availablePostprocessors).map((post) => (
          <option key={post} value={post}>
            {post}
          </option>
        ))}
      </select>
    </div>
  );
};
