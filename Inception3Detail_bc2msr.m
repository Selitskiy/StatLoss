%% Clear everything 
clear all; close all; clc;
ngpu = gpuDeviceCount();
for i=1:ngpu
    reset(gpuDevice(i));
end


%% Dataset root folder template and suffix
dataFolderTmpl = '~/data/BC2_Sfx';
dataFolderSfx = '1072x712';


% Create imageDataset of all images in selected baseline folders
[baseSet, dataSetFolder] = createBCbaselineIDS6b(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
trainingSet = baseSet;

% Count number of the classes ('stable' - presrvation of the order - to use
% later for building confusion matrix)
labels = unique(trainingSet.Labels, 'stable');
[nClasses, ~] = size(labels);

% Print image count for each label
%trainCountTable = countEachLabel(trainingSet);
%trainCount = sum(table2array(trainCountTable(:,2)), 1);
[trainCount, ~] = size(trainingSet.Files);

t1 = clock();
                        
%% Split Database into Training & Test Sets in the ratio 80% to 20% (usually comment out)
%[trainingSet, testSet] = splitEachLabel(baseSet, 0.4, 'randomize'); 

%% Swarm of models
nModels = 7;
myNets = [];
save_net_fileT = '~/data/in_swarm';
%save_s1net_fileT = '~/data/in_swarm1_sv';
save_s2net_fileT = '~/data/in_swarm2CL_sv';

for s=1:nModels
    n_ll = 315;
    % Load saved model if exists
    save_net_file = strcat(save_net_fileT, int2str(s), '.mat');
    if isfile(save_net_file)
        load(save_net_file, 'myNet');
    end
       
    if exist('myNet') == 0
        % Load Pre-trained Network (AlexNet)   
        % AlexNet is a pre-trained network trained on 1000 object categories. 
        %alex = alexnet;
        incept3 = inceptionv3;

        %% Review Network Architecture 
        lgraph_r = layerGraph(incept3);
        figure
        plot(lgraph_r)

        %% Modify Pre-trained Network 
        % AlexNet was trained to recognize 1000 classes, we need to modify it to
        % recognize just nClasses classes. 
        %n_ll = 315;

        lgraph = replaceLayer(lgraph_r, 'predictions', fullyConnectedLayer(nClasses, 'Name', 'predictions'));
        lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', classificationLayer('Name', 'ClassificationLayer_predictions'));

        % Perform Transfer Learning
        % For transfer learning we want to change the weights of the network 
        % ever so slightly. How much a network is changed during training is 
        % controlled by the learning rates. 
        opts = trainingOptions('adam',...
                       'ExecutionEnvironment','parallel',...
                       'InitialLearnRate', 0.001,...
                       'LearnRateSchedule', 'piecewise',...
                       'LearnRateDropPeriod', 5,...
                       'LearnRateDropFactor', 0.9,...
                       'MiniBatchSize', 128,...
                       'MaxEpochs', 10);
                   
                      %'Shuffle', 'every-epoch',... 
                      %'Plots', 'training-progress',...

        % Train the Network 
        % This process usually takes about 30 minutes on a desktop GPU. 
    
        % Shuffle training set for more randomness
        trainingSetS = shuffle(trainingSet);
    
        while (exist('TInfo')==0) || (TAcc < 90.) 
            [myNet, TInfo] = trainNetwork(trainingSetS, lgraph, opts);
            [~, TAccLast] = size(TInfo.TrainingAccuracy);
            TAcc = TInfo.TrainingAccuracy(TAccLast);
        end
        clear('TInfo');
        
        save(save_net_file, 'myNet');

    end
    
    myNets = [myNets, myNet];
    
    clear('myNet');
    clear('trainingSetS');
    clear('lgraph');
    clear('incept3');
    
end

Mem2 = zeros([ngpu 1]);
OMem2 = 0;
for i=1:ngpu
    dev = gpuDevice(i);
    Mem2(i) = dev.TotalMemory - dev.AvailableMemory;
    OMem2 = OMem2 + Mem2(i);
end
t2 = clock();
fprintf('Incept3 training N images:%d time:%.3f, models %d\n', trainCount, etime(t2, t1), nModels);


%% Mem cleanup
for i=1:ngpu
    reset(gpuDevice(i));
end


%% Traditional accuracy (usually comment out)
%predictedLabels = classify(myNet, testSet); 
%accuracy = mean(predictedLabels == testSet.Labels)

%predictedScores = predict(myNet, testSet);
%[nImages, ~] = size(predictedScores);
%for k=1:nImages
%    maxScore = 0;
%    maxScoreNum = 0;
%    maxScoreClass = "S";
%    correctClass = testSet.Labels(k);
%    for l=1:nClasses
%        if maxScore <= predictedScores(k, l)
%            maxScore = predictedScores(k, l);
%            maxScoreNum = l;
%            maxScoreClass = myNet.Layers(25).Classes(l);
%        end
%    end   
%    fprintf("%s %f %s \n", correctClass, maxScore, maxScoreClass);
%end



%% Reliability training datasets
% Create imageDataset vector of images in selected makeup folders
[testRSets, testRDataSetFolders] = createBCtestIDSvect6b1(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);

t1 = clock();

%% Create Matrix of Softmax Activations
[nMakeups, ~] = size(testRSets);
nImgsTot = 0;

for i=1:nMakeups
    [nImages, ~] = size(testRSets{i}.Files);
    nImgsTot = nImgsTot + nImages;
end

Act = zeros([nImgsTot nClasses nModels]);
Verd = zeros([nImgsTot nModels]);
Strong = zeros([nImgsTot nModels]);

Act2 = zeros([nImgsTot nModels]);

nEnsLabels = nModels + 1;
ActS = zeros([nImgsTot nClasses*nModels+nEnsLabels]);

nRealCLOut = nEnsLabels; %1;
nMem = 1000;
nCLOut = nRealCLOut + 1 + 2*nMem;
VerdS = zeros([nImgsTot nCLOut]);



%% Populate Matrix of Softmax Activations
nImgsCur = 1;
for i=1:nMakeups   
    [nImages, ~] = size(testRSets{i}.Files);
    
    fprintf('Makeup # %d/%d\n', i, nMakeups);    
            
    %% Walk through model Swarm
    ActPF = zeros([nImages nClasses nModels]);
    VerdPF = zeros([nImages nModels]);
    testRSet = testRSets{i};
    parfor s=1:nModels 
        
        predictedLabels = classify(myNets(s), testRSet);
        predictedScores = predict(myNets(s), testRSet);
        ActPF(:, :, s) = predictedScores;
        VerdPF(:, s) = (testRSet.Labels == predictedLabels);

    end
    Act(nImgsCur:nImgsCur + nImages - 1, :, :) = ActPF(:, :, :);        
    Verd(nImgsCur:nImgsCur + nImages - 1, :) = VerdPF(:, :);
    
    nImgsCur = nImgsCur + nImages;
    
end

%% Sorted activations of model candidates
[ActC, I] = sort(Act, 2, 'descend');


%% Collect the strongest softmax of models and flatten ensamble verdict vector
% (Another place to add supervisor to rank models based of its core)
%Strong(:, :) = mean(ActC(:, :, :), 2);
Strong(:, :) = ActC(:, 1, :);


%% Sort other models by their strongest softmax 
[StrongC, IStrong] = sort(Strong, 2, 'descend');

% And sort their softmax activations in the same way as activations
% in the model with strongest right softmax
for k=1:nImgsTot
    for si=1:nModels
        
        s = IStrong(k, si);
        %if si == 1   
            %if Verd(k, s) > 0
                %VerdS(k) = sum(Verd(k, :), 2);
            %end
        %end
        
        ActS(k, nClasses*(si-1)+1:nClasses*si) = Act(k, I(k, :, IStrong(k, 1)), s);

    end
    nCorrect =  sum(Verd(k, :), 2);
    VerdS(k, 1+nCorrect) = 1;
    ActS(k, nClasses*nModels+1:nClasses*nModels+nEnsLabels) = VerdS(k,1:nEnsLabels);
end


for i=1:ngpu
    reset(gpuDevice(i));
end

%% Train Supervisor model
save_s2net_file = strcat(save_s2net_fileT, int2str(nModels), '.mat');
%if isfile(save_s2net_file)
%    load(save_s2net_file, 'super2Net');
%else
%    clear('super2Net');
%end

Yt = categorical(VerdS');
[nDim, nVerdicts] =  size(countcats(Yt));
%VerdS( VerdS(:,1) > (nDim-1)/2., 2) = 1.;

%if exist('super2Net') == 0
    
    %Yt = categorical(VerdS');
    %[~, nVerdicts] =  size(countcats(Yt));

    nLayer1 = nClasses*nModels + nEnsLabels + 1;
    nLayer2 = 2*nClasses*nModels+2;
    nLayer3 = 2*nClasses*nModels+1;

    sLayers = [
        featureInputLayer(nClasses*nModels + nEnsLabels)
        fullyConnectedM1ReluLayer("L1", nClasses*nModels + nEnsLabels, nEnsLabels, nLayer1)
        fullyConnectedM1ReluLayer("L2", nLayer1, nEnsLabels, nLayer2)
        %fullyConnectedM1ReluLayer("L21", nLayer2, nEnsLabels, nLayer2)
        %fullyConnectedM1ReluLayer("L22", nLayer2, nEnsLabels, nLayer2)
        fullyConnectedCLSMLayer("L3", nLayer2, nEnsLabels, nRealCLOut, nCLOut, nEnsLabels-1, nMem)
        TCLSMRegression("L4", nEnsLabels, nMem)
    ];
    sgraph = layerGraph(sLayers);

    sOptions = trainingOptions('adam', ...
        'ExecutionEnvironment','auto',...
        'Shuffle', 'every-epoch',...
        'MiniBatchSize', 64, ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',500, ...
        'Verbose',true, ...
        'Plots','training-progress');

        %'LearnRateSchedule', 'piecewise',...
        %'LearnRateDropPeriod', 5,...
        %'LearnRateDropFactor', 0.9,...
    
    super2Net = trainNetwork(ActS, VerdS, sgraph, sOptions);


    % Verify
    supervisorPredictedScores = predict(super2Net, ActS);
    
    %save(save_s2net_file, 'super2Net');

%end

t2 = clock();
fprintf('Incept3 Supervisor training N images:%d time:%.3f, models %d\n', nImgsTot, etime(t2, t1), nModels);


%% Makeup datasets
mkDataSetFolder = strings(0);
mkLabel = strings(0);

% Create imageDataset vector of images in selected makeup folders
[testSets, testDataSetFolders] = createBCtestIDSvect6b(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);


%%
[nMakeups, ~] = size(testSets);

mkTable = cell(nMakeups, nClasses+4);

%%

% Write per-image scores to a file
fd = fopen( strcat('predict_in_6bmsrTr',int2str(nModels),'.txt'),'w' );

fprintf(fd, "CorrectClass MaxScore MaxScoreClass TrustScore TrustThresholdNorm TrustFlag05 TrustFlagTr FileName");
for l=1:nClasses
    fprintf(fd, " %s", myNets(1).Layers(n_ll).Classes(l));
end
fprintf(fd, "\n");




%% Create Matrix of Softmax Activations
[nMakeups, ~] = size(testSets);

%DEBUG
%nMakeups = 10;

nImgsTot = 0;

for i=1:nMakeups
    [nImages, ~] = size(testSets{i}.Files);
    nImgsTot = nImgsTot + nImages;
end

fprintf('Incept3 test N images:%d, models %d\n', nImgsTot, nModels);

ActT = zeros([nImgsTot nClasses nModels]);
VerdT = zeros([nImgsTot nModels]);
StrongT = zeros([nImgsTot nModels]);
ActTS = zeros([nImgsTot nClasses*nModels+1]);
VerdTS = zeros([nImgsTot nCLOut]);

%% Populate Matrix of Softmax Activations
predictedLabelsSwarm = cell(nMakeups, nModels);
predictedScoresSwarm = cell(nMakeups, nModels);
nImgsCur = 1;
for i=1:nMakeups   
    [nImages, ~] = size(testSets{i}.Files);
    
    fprintf('Makeup # %d/%d\n', i, nMakeups);
            
    %% Walk through model Swarm
    ActTPF = zeros([nImages nClasses nModels]);
    VerdTPF = zeros([nImages nModels]);
    testSet = testSets{i};
    parfor s=1:nModels 
        % Test main network performance
        predictedLabels = classify(myNets(s), testSets{i});
        predictedLabelsSwarm{i, s} = predictedLabels;
        predictedScores = predict(myNets(s), testSets{i});
        predictedScoresSwarm{i, s} = predictedScores;
        
        ActTPF(:, :, s) = predictedScores;        
        VerdTPF(:, s) = (testSet.Labels == predictedLabels);
    end
    ActT(nImgsCur:nImgsCur + nImages - 1, :, :) = ActTPF(:, :, :);
    VerdT(nImgsCur:nImgsCur + nImages - 1, :) = VerdTPF(:, :);

    nImgsCur = nImgsCur + nImages;
end

%% Sorted activations of model candidates
[ActTC, IT] = sort(ActT, 2, 'descend');

%for s=1:nModels
%    super2Scores = predict(mySuper1Nets(s), ActTC(:, :, s));
%    Act2T(:, s) = super2Scores(:, 2);
%end        

%[LikelyTC, ILikelyT] = sort(Act2T, 2, 'descend');


% Collect the strongest softmax of models and flatten ensamble verdict vector
% (Another place to add supervisor to rank models based of its core)
%StrongT(:, :) = mean(ActTC(:, :, :), 2);
StrongT(:, :) = ActTC(:, 1, :);

%% Sort other models by their strongest softmax 
[StrongTC, IStrongT] = sort(StrongT, 2, 'descend');

% And sort their softmax activations in the same way as activations
% in the model with strongest right softmax
for k=1:nImgsTot
      
    for si=1:nModels
        
        s = IStrongT(k, si);
        ActTS(k, nClasses*(si-1)+1:nClasses*si) = ActT(k, IT(k, :, IStrongT(k, 1)), s);
        
    end
end   

   
%%
% Supervisor network
%supervisorPredictedLabels = classify(super2Net, ActTS); 
supervisorPredictedScores = predict(super2Net, ActTS);
sLayers = super2Net.Layers;
sgraph = layerGraph(sLayers);

nImgsCur = 0;
for i=1:nMakeups 
    clear('predictedScoresS');
    clear('predictedLabelsS');
    
    for s=1:nModels 
        predictedScoresS(:, :, s) = predictedScoresSwarm{i, s};
        predictedLabelsS(:, s) = predictedLabelsSwarm{i, s};
    end
    % Ensemble voting
    [~, MI] = max(countcats(predictedLabelsS, 2), [], 2);
    predictedLabelsCat = (categories(predictedLabelsS));
    predictedLabels = predictedLabelsCat(MI);
    
    predictedScores = mean(predictedScoresS, 3);
        
    [nImages, ~] = size(testSets{i}.Files);
    for k=1:nImages
    
        maxScore = 0;
        maxScoreNum = 0;
        maxScoreClass = "S";
        correctClass = testSets{i}.Labels(k);
        for l=1:nClasses
            if maxScore <= predictedScores(k, l)
                maxScore = predictedScores(k, l);
                maxScoreNum = l;
                maxScoreClass = predictedLabels{k};
            end
        end

        fprintf(fd, "%s %f %s %f %f %d %d %s", correctClass, maxScore, maxScoreClass,...
            supervisorPredictedScores(nImgsCur+k, 1),...
            supervisorPredictedScores(nImgsCur+k, 2),...
            supervisorPredictedScores(nImgsCur+k, 1) > (nDim-1)/2.,...
            supervisorPredictedScores(nImgsCur+k, 1) > supervisorPredictedScores(nImgsCur+k, 2),...
            testSets{i}.Files{k});
        for l=1:nClasses
            fprintf(fd, " %f", predictedScores(k, l));
        end
        fprintf(fd, "\n");
        
    end
    
    nImgsCur = nImgsCur + nImages;
    
    [tmpStr, ~] = strsplit(testSets{i}.Files{1}, '/');
    fprintf("%s", tmpStr{1,7}); 
    mean(predictedScores)
    
    
    %% Compute average accuracy
    meanMkAcc = mean(predictedLabels == testSets{i}.Labels);
    mkTable{i,1} = testDataSetFolders(i);
    mkTable{i,2} = meanMkAcc;
    
    %%
    [tn, ~] = size(testSets{i}.Files);
    
    meanMkConf = zeros(1, nClasses);

    maxAccCat = '';
    maxAcc = 0;
    
    %%    
    j = 1;   
    for j = 1:nClasses

        tmpStr = strings(tn,1);
        tmpStr(:) = string(labels(j));
    
        meanMkConf(j) = mean(string(predictedLabels) == tmpStr);
        mkTable{i, 4+j} = meanMkConf(j);
        
        %find the best category match
        if maxAcc <= meanMkConf(j)
            maxAccCat = tmpStr(j);
            maxAcc = meanMkConf(j);
        end
        
    end
    mkTable{i,3} = maxAccCat;
    mkTable{i,4} = maxAcc;
    
end

%% Results
varNames = cellstr(['TestFolder' 'Accuracy' 'BestGuess' 'GuessScore' string(labels)']);
cell2table(mkTable, 'VariableNames', varNames)

fclose(fd);
