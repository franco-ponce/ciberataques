% Custom layer: reshapes a flat feature vector into an image tensor (used in the decoder)
classdef reshapeLayer2 < nnet.layer.Layer & nnet.layer.Formattable
    properties
        TargetSize  % output size [H, W, C] (without batch)
    end
    methods
        % Constructor
        function layer = reshapeLayer2(targetSize, name)
            layer.Name = name;  % layer name in the network
            layer.Description = "Reshape to [" + join(string(targetSize), ", ") + "]";  % text shown in analyzeNetwork
            %layer.Description = ['Reshape to [', strjoin(...), ']']
            layer.Type = 'Reshape';
            layer.TargetSize = targetSize;
        end
        % Forward pass: [Features x Batch] -> [H x W x C x Batch]
        function Z = predict(layer, X)
            if ismatrix(X)
                N = size(X, 2);  % batch size (input as 'CB')
                Zraw = reshape(X, [layer.TargetSize, N]);
            else
                Zraw = reshape(X, [layer.TargetSize, size(X, 4)]);  % input already 4-D
            end
            Z = dlarray(Zraw, 'SSCB');  % label dims: spatial, spatial, channel, batch
        end
        % Backward pass: undo the reshape on the gradient
        function dLdX = backward(layer, X, ~, dLdZ, ~)
            dLdX = reshape(dLdZ, size(X));  % gradient back to the input shape
        end
    end
end
