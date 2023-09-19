classdef Incept3Layers

    properties

    end

    methods
        function net = Incept3Layers()            
        end


        function net = Create(net)

            % Load Pre-trained Network
            incept3 = inceptionv3;

            %% Review Network Architecture 
            lgraph_r = layerGraph(incept3);
            %figure
            %plot(lgraph_r)

            %% Modify Pre-trained Network 

            lgraph = replaceLayer(lgraph_r, 'predictions', fullyConnectedLayer(net.n_out, 'Name', 'predictions'));
            lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', classificationLayer('Name', 'ClassificationLayer_predictions'));

    
            net.lGraph = lgraph;

            net.options = trainingOptions('adam', ...
                'ExecutionEnvironment','auto',... %'parallel',...
                'Shuffle', 'every-epoch',...
                'MiniBatchSize', net.mb_size, ...
                'InitialLearnRate', net.ini_rate, ...
                'MaxEpochs',net.max_epoch);

                %'LearnRateSchedule', 'piecewise',...
                %'LearnRateDropPeriod', 5,...
                %'LearnRateDropFactor', 0.9,...
                %'Plots', 'training-progress',...
        end


    end

end
