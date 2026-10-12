classdef reshapeLayer < nnet.layer.Layer
    % Custom reshape layer for decoder
    
    properties (Learnable)
    end
    
    properties
        OutputSize
    end
    
    methods
        function layer = reshapeLayer(outputSize, varargin)
            % Constructor
            layer.OutputSize = outputSize;
            layer.Name = 'reshape';
            % FIX: Use string concatenation compatible with all MATLAB versions
            layer.Description = ['Reshape to ', mat2str(outputSize)];
            
            % Parse optional name-value pairs
            for i = 1:2:length(varargin)
                if strcmpi(varargin{i}, 'Name')
                    layer.Name = varargin{i+1};
                end
            end
        end
        
        function Z = predict(layer, X)
            % Forward pass
            outputSize = layer.OutputSize;
            batchSize = size(X, 2);
            
            % Reshape from [Features × Batch] to [H × W × C × Batch]
            Z = reshape(X, [outputSize, batchSize]);
        end
    end
end
