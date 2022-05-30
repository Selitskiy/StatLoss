classdef TCLmRegression < nnet.layer.RegressionLayer
        
    properties
        % (Optional) Layer properties.

        % Layer properties go here.

        % Number of last observations in memory
        numLabelChannels
        MMax
    end
 
    methods
        function layer = TCLmRegression(name, numLabelChannels, MMax)           
            % (Optional) Create a myRegressionLayer.

            % Layer constructor function goes here.

            % Set layer name.
            layer.Name = name;

            layer.numLabelChannels = numLabelChannels;
            layer.MMax = MMax;

            %Initialize Memory (columns 1-Regression value (R), 2-Threshold, 3-Lable(L))

        end

        
        function loss = forwardLoss(layer, Y, T)
            % Return the loss between the predictions Y and the training
            % targets T.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         loss  - Loss between Y and T

            % Layer forward loss function goes here.
            
            % calculate channel-wise (row) errors
            %u,Tr
            [m, r, n] = size(Y);
            %fprintf('fwd loss Y m=%d r=%d n=%d\n', m, r, n);  m-channels, r-observation, n=1

            %Le
            [mt, rt, nt] = size(T);
            %fprintf('fwd loss T mt=%d rt=%d nt=%d\n', mt, rt, nt);  %m-channels, r-observation, n=1

            %errSquares1 = (Y - T) .^ 2;
            errSquares1 = (Y(1:layer.numLabelChannels,:) - T(1:layer.numLabelChannels,:)) .^ 2;

            errSquares2 = zeros([m, r], "like", T);
            % Li < T, Yi > T or Li >T, Yi<T
            C = (Y(layer.numLabelChannels+2:layer.numLabelChannels+1+layer.MMax,:) > Y(2,1)) & (Y(layer.numLabelChannels+2+layer.MMax:layer.numLabelChannels+1+2*layer.MMax,:) < Y(2,1));
            D = (Y(layer.numLabelChannels+2:layer.numLabelChannels+1+layer.MMax,:) < Y(2,1)) & (Y(layer.numLabelChannels+2+layer.MMax:layer.numLabelChannels+1+2*layer.MMax,:) > Y(2,1));
            E = C | D;
            errSquares2( E ) = (Y(E) - Y(2,1)) .^ 2;
            
            % Number of wrong predictions
            fpfn = sum(E,'all');

            %summarise parallel observations (minibatches)
            % Take mean over mini-batch
            loss1 = sum(errSquares1, 'all') / m / r;
            loss2 = sum(errSquares2, 'all') / m / r;

            fprintf('fwd loss CL=%f l1=%f l2=%f Y1=%f Y2=%f T1=%f T2=%f\n', fpfn/r/m ,loss1, loss2, Y(1,1), Y(2,1), T(1,1), T(2,1));

            loss = loss1 + loss2;

            %errSquares1 = (Y(1,:) - T(1,:)) .^ 2;

            %errSquares2 = zeros([m, r], "like", T);
            %% Li < T, Yi > T or Li >T, Yi<T
            %C = (Y(2+1:2+layer.MMax,:) > Y(2,1)) & (Y(2+1+layer.MMax:2+2*layer.MMax,:) < Y(2,1));
            %D = (Y(2+1:2+layer.MMax,:) < Y(2,1)) & (Y(2+1+layer.MMax:2+2*layer.MMax,:) > Y(2,1));
            %E = C | D;
            %errSquares2( E ) = (Y(E) - Y(2,1)) .^ 2;
            
            % Number of wrong predictions
            %fpfn = sum(E,'all');

            %summarise parallel observations (minibatches)
            % Take mean over mini-batch
            %loss1 = sum(errSquares1, 'all') / m / r;
            %loss2 = sum(errSquares2, 'all') / m / r;

            %fprintf('fwd loss CL=%f l1=%f l2=%f Y1=%f Y2=%f T1=%f T2=%f\n', fpfn/r/m ,loss1, loss2, Y(1,1), Y(2,1), T(1,1), T(2,1));

            %loss = loss1;% + loss2;

        end
        

        function dLdY = backwardLoss(layer, Y, T)
            % (Optional) Backward propagate the derivative of the loss 
            % function.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the 
            %                 predictions Y        

            % Layer backward loss function goes here.

            % calculate channel-wise (row) error dervivatives
            %u,Tr
            [m, r, n] = size(Y);
            %fprintf('back loss Y m=%d r=%d n=%d\n', m, r, n);  %n=1
            %Le
            [mt, rt, nt] = size(T);
            %fprintf('back loss T m2=%d r2=%d n2=%d\n', m2, r2, n2);  %n=1

            %errSquareDer = 2. * (Y - T);
            errSquareDer = zeros([m, r], "like", T);
            errSquareDer(1:layer.numLabelChannels,:) = 2. * (Y(1:layer.numLabelChannels,:) - T(1:layer.numLabelChannels,:));

            
            % Li < T, Yi > T or Li >T, Yi<T
            C = (Y(layer.numLabelChannels+2:layer.numLabelChannels+1+layer.MMax,:) > Y(2,1)) & (Y(layer.numLabelChannels+2+layer.MMax:layer.numLabelChannels+1+2*layer.MMax,:) < Y(2,1));
            D = (Y(layer.numLabelChannels+2:layer.numLabelChannels+1+layer.MMax,:) < Y(2,1)) & (Y(layer.numLabelChannels+2+layer.MMax:layer.numLabelChannels+1+2*layer.MMax,:) > Y(2,1));
            E = C | D;
            errSquareDer(2, :) = - 2. * sum(Y(E) - Y(2,1), 1);


            %fprintf('back loss Tt dldu=%f dldt=%f\n', errSquareDer(1,1), errSquareDer(2,1));

            dLdY = errSquareDer;


            %errSquareDer = zeros([m, r], "like", T);
            %errSquareDer(1,:) = 2. * (Y(1,:) - T(1,:));
            
            %% Li < T, Yi > T or Li >T, Yi<T
            %C = (Y(2+1:2+layer.MMax,:) > Y(2,1)) & (Y(2+1+layer.MMax:2+2*layer.MMax,:) < Y(2,1));
            %D = (Y(2+1:2+layer.MMax,:) < Y(2,1)) & (Y(2+1+layer.MMax:2+2*layer.MMax,:) > Y(2,1));
            %E = C | D;
            %errSquareDer(2, :) = - sum(Y(E) - Y(2,1), 1);

            %fprintf('back loss Tt dldu=%f dldt=%f\n', errSquareDer(1,1), errSquareDer(2,1));

            %dLdY = errSquareDer;

            %[m2, r2, n2] = size(dLdY);
            %fprintf('back loss dLdY m2=%d r2=%d n2=%d\n', m2, r2, n2);  %n=1
        end

    end
end