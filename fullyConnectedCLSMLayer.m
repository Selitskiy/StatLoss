classdef fullyConnectedCLSMLayer < nnet.layer.Layer %& nnet.layer.Formattable

    properties
        % (Optional) Layer properties.

        % Declare layer properties here.

        % Number input channels
        numInChannels
        numLabelChannels
        numRealOutChannels
        numOutChannels
        MMax
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Declare learnable parameters here.
        Tr
        
        % Weights
        W
        %Bias
        W0
    end

    properties (State)
        % (Optional) Layer state parameters.

        % Declare state parameters here.
        M
        MFill
        MCurr
    end

    %properties (Learnable, State)
        % (Optional) Nested dlnetwork objects with both learnable
        % parameters and state.

        % Declare nested networks with learnable and state parameters here.
    %end

    methods
        function layer = fullyConnectedCLSMLayer(name, numInChannels, numLabelChannels, numRealOutChannels, numOutChannels, lMax, MMax)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Define layer constructor function here.

            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "FullyConnectedCLSM" + numInChannels + " channels";

            layer.numInChannels = numInChannels;
            layer.numLabelChannels = numLabelChannels;
            layer.numRealOutChannels = numRealOutChannels;
            layer.numOutChannels = numOutChannels;


            % Initialize weight coefficients.
            layer.W = rand([layer.numOutChannels, layer.numInChannels-layer.numLabelChannels]);

            layer.W0 = rand([layer.numOutChannels, 1]);

            % Initialize trhreshold
            layer.Tr = lMax * 0.2;%rand() * lMax;

            % Initialize memory parameters
            layer.MMax = MMax;


            % Initialize layer states.
            %layer = resetState(layer);

            layer.M = zeros([numRealOutChannels+2, layer.MMax]);
            layer.MFill = 0;
            layer.MCurr = 0;

            %global Mstate;
            %Mstate = zeros([2, layer.MMax]);
            %global MstateFill;
            %MstateFill = 0;
            %global MstateCurr;
            %MstateCurr = 0;

        end

        function [Z, M, MFill, MCurr] = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result and updated state.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Input data
            % Outputs:
            %         Z     - Output of layer forward function
            %         state - (Optional) Updated layer state.
            %
            %  - For layers with multiple inputs, replace X with X1,...,XN, 
            %    where N is the number of inputs.
            %  - For layers with multiple outputs, replace Z with 
            %    Z1,...,ZM, where M is the number of outputs.
            %  - For layers with multiple state parameters, replace state 
            %    with state1,...,stateK, where K is the number of state 
            %    parameters.

            % Define layer predict function here.
            % c - channels, n - observations
            [c, n] = size(X);

            %Z = layer.W * X + layer.W0;
            Z = layer.W * X(1:layer.numInChannels-layer.numLabelChannels, :) + layer.W0;
            Z = softmax(Z(1:layer.numRealOutChannels, :), 'DataFormat','CB');

            Z(layer.numRealOutChannels+1, :) = repmat(layer.Tr, [1, n]);
            Z(layer.numRealOutChannels+2:layer.numRealOutChannels+1+layer.MMax, :) = repmat(layer.M(1, :)', [1, n]);
            Z(layer.numRealOutChannels+2+layer.MMax:layer.numRealOutChannels+1+2*layer.MMax, :) = repmat(layer.M(layer.numRealOutChannels+2, :)', [1, n]);

            M = layer.M;
            MFill = layer.MFill;
            MCurr = layer.MCurr;

            %global MstateFill;
            %global MstateCurr;
            %global Mstate;
            %for i=1:n
            %    if(MstateFill < layer.MMax)
            %        MstateFill = (MstateFill + 1);
            %        MFill = MstateFill;
            %    end
            %    MstateCurr = mod(MstateCurr, layer.MMax);
            %    Mstate(:, MstateCurr+1) = Z(:, i);
            %    M(:, MstateCurr+1) = Z(:, i);
            %    fprintf('state curr=%d Z1=%f Z2=%f\n', MstateCurr, Mstate(1, MstateCurr+1), Mstate(2, MstateCurr+1));
            %    MstateCurr = MstateCurr+1;
            %    MCurr = MstateCurr;
            %end

            for i=1:n
                if(MFill < layer.MMax)
                    MFill = (MFill + 1);
                end
                MCurr = mod(MCurr, layer.MMax);
                M(1:layer.numRealOutChannels+1, MCurr+1) = Z(1:layer.numRealOutChannels+1, i);
                M(layer.numRealOutChannels+2, MCurr+1) = X(layer.numInChannels, i);
                %fprintf('state curr=%d Z1=%f Z2=%f\n', MCurr, M(1, MCurr+1), M(2, MCurr+1));
                MCurr = MCurr+1;
            end

            %fprintf('state c=%d n=%d MtFill=%d\n', c, n, MtFill);
        end

        %function [Z,state,memory] = forward(layer,X)
            % (Optional) Forward input data through the layer at training
            % time and output the result, updated state, and a memory
            % value.
            %
            % Inputs:
            %         layer - Layer to forward propagate through 
            %         X     - Layer input data
            % Outputs:
            %         Z      - Output of layer forward function 
            %         state  - (Optional) Updated layer state 
            %         memory - (Optional) Memory value for custom backward
            %                  function
            %
            %  - For layers with multiple inputs, replace X with X1,...,XN, 
            %    where N is the number of inputs.
            %  - For layers with multiple outputs, replace Z with 
            %    Z1,...,ZM, where M is the number of outputs.
            %  - For layers with multiple state parameters, replace state 
            %    with state1,...,stateK, where K is the number of state 
            %    parameters.

            % Define layer forward function here.
        %end

        %function layer = resetState(layer)
            % (Optional) Reset layer state.

            % Define reset state function here.
        %    layer.M = zeros([2, layer.MMax]);
        %    layer.MFill = 0;

        %    global Mstate;
        %    Mstate = zeros([2, layer.MMax]);
        %    global MstateFill;
        %    MstateFill = 0;
        %    global MstateCurr;
        %    MstateCurr = 0;
        %end

        %function [dLdX,dLdW,dLdSin] = backward(layer,X,Z,dLdZ,dLdSout,memory)
            % (Optional) Backward propagate the derivative of the loss
            % function through the layer.
            %
            % Inputs:
            %         layer   - Layer to backward propagate through 
            %         X       - Layer input data 
            %         Z       - Layer output data 
            %         dLdZ    - Derivative of loss with respect to layer 
            %                   output
            %         dLdSout - (Optional) Derivative of loss with respect 
            %                   to state output
            %         memory  - Memory value from forward function
            % Outputs:
            %         dLdX   - Derivative of loss with respect to layer input
            %         dLdW   - (Optional) Derivative of loss with respect to
            %                  learnable parameter 
            %         dLdSin - (Optional) Derivative of loss with respect to 
            %                  state input
            %
            %  - For layers with state parameters, the backward syntax must
            %    include both dLdSout and dLdSin, or neither.
            %  - For layers with multiple inputs, replace X and dLdX with
            %    X1,...,XN and dLdX1,...,dLdXN, respectively, where N is
            %    the number of inputs.
            %  - For layers with multiple outputs, replace Z and dlZ with
            %    Z1,...,ZM and dLdZ,...,dLdZM, respectively, where M is the
            %    number of outputs.
            %  - For layers with multiple learnable parameters, replace 
            %    dLdW with dLdW1,...,dLdWP, where P is the number of 
            %    learnable parameters.
            %  - For layers with multiple state parameters, replace dLdSin
            %    and dLdSout with dLdSin1,...,dLdSinK and 
            %    dLdSout1,...dldSoutK, respectively, where K is the number
            %    of state parameters.

            % Define layer backward function here.
        %end
    end
end