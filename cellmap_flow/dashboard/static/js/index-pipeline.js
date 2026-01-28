import React from 'react';
import ReactDOM from 'react-dom';
import { ReactFlowProvider } from 'reactflow';
import { PipelineBuilder } from './pipeline-builder';

// This will be injected by Flask
const APP_CONFIG = window.APP_CONFIG || {
  availableNormalizers: {},
  availablePostprocessors: {},
};

ReactDOM.render(
  <ReactFlowProvider>
    <PipelineBuilder
      availableNormalizers={APP_CONFIG.availableNormalizers}
      availablePostprocessors={APP_CONFIG.availablePostprocessors}
    />
  </ReactFlowProvider>,
  document.getElementById('pipeline-root')
);
