%% Clear everything 
clear all; close all; clc;
ngpu = gpuDeviceCount();
for i=1:ngpu
    reset(gpuDevice(i));
end


%% Dataset root folder template and suffix
dataFolderTmpl = '~/data/BC2_Sfx';
dataFolderSfx = '1072x712';

kfold_pref = "";


% Create imageDataset of all images in selected baseline folders
[baseSet, dataSetFolder] = createBCbaselineIDS6b1(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
%[baseSet, dataSetFolder] = createBCbaselineIDS6b2(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
%[baseSet, dataSetFolder] = createBCbaselineIDS6b3(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
%[baseSet, dataSetFolder] = createBCbaselineIDS6b4(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
%[baseSet, dataSetFolder] = createBCbaselineIDS6b5(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);

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

mb_size = 128;

for s=1:nModels
    n_ll = 315;
    % Load saved model if exists
    save_net_file = strcat(save_net_fileT, int2str(s), kfold_pref, '.mat');
    if isfile(save_net_file)
        load(save_net_file, 'myNet');
    end
       
    if exist('myNet') == 0
        % Load Pre-trained Network
        incept3 = inceptionv3;

        %% Review Network Architecture 
        lgraph_r = layerGraph(incept3);
        %figure
        %plot(lgraph_r)

        %% Modify Pre-trained Network 

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
                       'MiniBatchSize', mb_size,...
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


%% Traditional accuracy (usually comment out) DEBUG
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
%[testRSets, testRDataSetFolders] = createBCtestIDSvect6b2(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
%[testRSets, testRDataSetFolders] = createBCtestIDSvect6b3(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
%[testRSets, testRDataSetFolders] = createBCtestIDSvect6b4(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
%[testRSets, testRDataSetFolders] = createBCtestIDSvect6b5(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);

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

dimLabel = 1;
ActS = zeros([nImgsTot nClasses*nModels+dimLabel]);

nEnsLabels = nModels + 1;
% regression dimension (same as dimLabel here)
nRealCLOut = dimLabel;
% threwshold dimansion
nTrOut = 1;
% size of the last predictions table (state to use in loss calculation) 
nMem = 8192;
% technical output between state layere and loss
nCLOut = nRealCLOut + nTrOut + 2*nMem;

VerdS = zeros([nImgsTot nCLOut]);

mb_sizeS = 64;


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
%[ActC, I] = sort(Act, 2, 'descend');


%% Collect the strongest softmax of models and flatten ensamble verdict vector
%Strong(:, :) = ActC(:, 1, :);


%% Sort other models by their strongest softmax 
%[StrongC, IStrong] = sort(Strong, 2, 'descend');

% And sort their softmax activations in the same way as activations
% in the model with strongest right softmax
%for k=1:nImgsTot
%    for si=1:nModels
        
%        s = IStrong(k, si);
        
%        ActS(k, nClasses*(si-1)+1:nClasses*si) = Act(k, I(k, :, IStrong(k, 1)), s);

%    end

%    VerdS(k,1) = sum(Verd(k, :), 2);
%    ActS(k, nClasses*nModels+1) = VerdS(k,1);

%end
[ActS, VerdS] = makeUSDstrong(Act, Verd, ActS, VerdS, nClasses, 0, nImgsTot, nModels, dimLabel);


for i=1:ngpu
    reset(gpuDevice(i));
end

%% Train Supervisor model
save_s2net_file = strcat(save_s2net_fileT, int2str(nModels), kfold_pref, '.mat');
%if isfile(save_s2net_file)
%    load(save_s2net_file, 'super2Net');
%else
%    clear('super2Net');
%end

Yt = categorical(VerdS');
[nDim, nVerdicts] =  size(countcats(Yt));

%if exist('super2Net') == 0

    nLayer1 = nClasses*nModels+1+dimLabel;
    nLayer2 = 2*nClasses*nModels+1+dimLabel;
    nLayer3 = 2*nClasses*nModels+1+dimLabel;


        sOptions = trainingOptions('adam', ...
        'ExecutionEnvironment','auto',...
        'Shuffle', 'every-epoch',...
        'MiniBatchSize', mb_sizeS, ...
        'InitialLearnRate',0.01, ...
        'MaxEpochs',200, ...
        'Verbose',true, ...
        'Plots','training-progress');

        %'LearnRateSchedule', 'piecewise',...
        %'LearnRateDropPeriod', 5,...
        %'LearnRateDropFactor', 0.9,...


    sLayers = [
        featureInputLayer(nClasses*nModels+dimLabel)
        fullyConnectedM1ReluLayer("L1", nClasses*nModels+dimLabel, dimLabel, nLayer1)
        fullyConnectedM1ReluLayer("L2", nLayer1, dimLabel, nLayer2)
        %%fullyConnectedM1ReluLayer("L21", nLayer2, nLayer2)
        %%fullyConnectedM1ReluLayer("L22", nLayer2, nLayer2)
        fullyConnectedCLLayer("L3", nLayer2, dimLabel, nRealCLOut, nCLOut, nEnsLabels, nMem)
        TCLmRegression("L4", dimLabel, nMem)
    ];
    sgraph = layerGraph(sLayers);
    
    super2Net = trainNetwork(ActS, VerdS, sgraph, sOptions);


    % Verify
    %supervisorPredictedScorest = predict(super2Nett, ActSt);
    %rmset = sqrt( sum(( (supervisorPredictedScorest - VerdSt) .^ 2), 'all') );

    supervisorPredictedScores = predict(super2Net, ActS);
    rmse = sqrt( sum(( (supervisorPredictedScores - VerdS) .^ 2), 'all') );
    TrTrain = supervisorPredictedScores(1, 2);

    %rmsv = sqrt( sum(( (supervisorPredictedScores - supervisorPredictedScorest) .^ 2), 'all') );

    
    %save(save_s2net_file, 'super2Net');

%end

t2 = clock();
fprintf('Incept3 Supervisor training N images:%d time:%.3f, models %d\n', nImgsTot, etime(t2, t1), nModels);


%% Makeup datasets
mkDataSetFolder = strings(0);
mkLabel = strings(0);

% Create imageDataset vector of images in selected makeup folders
%[testSets, testDataSetFolders] = createBCtestIDSvect6b(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);
[testSets, testDataSetFolders] = createBCtestIDSvect6bWhole(dataFolderTmpl, dataFolderSfx, @readFunctionTrainIN_n);

%%
[nMakeups, ~] = size(testSets);

mkTable = cell(nMakeups, nClasses+4);

%% Continous and Active and MCMC-like Active flags
contF = 1;
c_pref = '';
if contF
    c_pref = strcat(c_pref, 'c');
end

nOrcaleLimit = 0; %0.001;
if nOrcaleLimit > 0
    c_pref = strcat(c_pref, num2str(nOrcaleLimit), 'a');
end

mcmcF = 0;
if mcmcF
    c_pref = strcat(c_pref, 'm');
end


% Write per-image scores to a file
fd = fopen( strcat('predict_in_6bmsrTr',int2str(nModels), kfold_pref, c_pref, '.txt'),'w' );

fprintf(fd, "CorrectClass MeanPredictScore PredictClassMax PredictClassTr TrustScore TrustThresholdTr TrustThresholdCurr TrustFlag05 TrustFlagTr TrustFlagCurr TrustScoreLb FileName");
for l=1:nClasses
    fprintf(fd, " %s", myNets(1).Layers(n_ll).Classes(l));
end
fprintf(fd, "\n");


%%
aOpts = trainingOptions('adam',...
    'ExecutionEnvironment','parallel',...
    'InitialLearnRate', 0.001,...
    'LearnRateSchedule', 'piecewise',...
    'LearnRateDropPeriod', 5,...
    'LearnRateDropFactor', 0.9,...
    'MiniBatchSize', mb_size,...
    'MaxEpochs', 5);

% Active re-training
[nTrainImg,~] = size(trainingSet.Files);
activeI = randperm(nTrainImg, mb_size);
activeFiles = trainingSet.Files(activeI);
activeLabels = trainingSet.Labels(activeI);
activeSet = imageDatastore(activeFiles); 
activeSet.Labels = activeLabels;
activeSet.ReadFcn = trainingSet.ReadFcn;

% Continous re-training
contI = randperm(nImgsTot, mb_sizeS);
contActTS = zeros([mb_sizeS nClasses*nModels+dimLabel]);
contVerdTS = zeros([mb_sizeS nCLOut]);
contActTS = ActS(contI, :);
contVerdTS = VerdS(contI, :);


%% Create Matrix of Softmax Activations
[nMakeups, ~] = size(testSets);

nImgsTotT = 0;

for i=1:nMakeups
    [nImages, ~] = size(testSets{i}.Files);
    nImgsTotT = nImgsTotT + nImages;
end

fprintf('Incept3 test N images:%d, models %d\n', nImgsTotT, nModels);

ActT = zeros([nImgsTotT nClasses nModels]);
VerdT = zeros([nImgsTotT nModels]);
StrongT = zeros([nImgsTotT nModels]);
ActTS = zeros([nImgsTotT nClasses*nModels+dimLabel]);
VerdTS = zeros([nImgsTotT nCLOut]);


% Supervisor network
supervisorPredictedScores = zeros([nImgsTotT nCLOut]);
cOptions = trainingOptions('adam', ...
    'ExecutionEnvironment','auto',...
    'Shuffle', 'every-epoch',...
    'MiniBatchSize', mb_sizeS, ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',10, ...
    'Verbose',true);%, ...
    %'Plots','training-progress');

%% Populate Matrix of Softmax Activations
predictedLabelsSwarm = cell(nMakeups, nModels);
predictedScoresSwarm = cell(nMakeups, nModels);
nImgsCur = 1;

activeCurI = 1;
nOracleAnsw = 0;
difPredLabelMax = 0;

contCurI = 1;


for i=1:nMakeups   
    [nImages, ~] = size(testSets{i}.Files);
    
    fprintf('Makeup # %d/%d\n', i, nMakeups);
        
    ActTPF = zeros([nImages nClasses nModels]);
    VerdTPF = zeros([nImages nModels]);
    testSet = testSets{i};
    
    for j=1:nImages
        fprintf('Image # %d/%d of Makeup %d/%d\n', j, nImages, i, nMakeups);
        img = readimage(testSet, j);
        testLabels = testSet.Labels(j);

        %% Walk through model Swarm
        parfor s=1:nModels 
            % Test main network performance
            predictedLabels = classify(myNets(s), img);
            predictedLabelsSwarm{i, s} = predictedLabels;
            predictedScores = predict(myNets(s), img);
            predictedScoresSwarm{i, s} = predictedScores;
        
            ActTPF(j, :, s) = predictedScores;        
            VerdTPF(j, s) = (testLabels == predictedLabels);
        end
        k = nImgsCur + j - 1;
        ActT(k, :, :) = ActTPF(j, :, :);
        VerdT(k, :) = VerdTPF(j, :);

        % Make Uncertainty Shape Descriptor
        [ActTS, VerdTS] = makeUSDstrong(ActT, VerdT, ActTS, VerdTS, nClasses, k, 0, nModels, dimLabel);
        supervisorPredictedScores(k,:) = predict(super2Net, ActTS(k, :));

        % Find label with number of occurance as predicted by superNet
        % Ensemble voting
        for s=1:nModels 
            predictedScoresS(:, :, s) = predictedScoresSwarm{i, s};
            predictedLabelsS(:, s) = predictedLabelsSwarm{i, s};
        end
        nCatMatches = countcats(predictedLabelsS, 2);
        [MaxCountLabels, MI] = max(nCatMatches, [], 2);
        predictedLabelsCat = (categories(predictedLabelsS));
        predictedLabels = predictedLabelsCat(MI);
    
        predictedScores = mean(predictedScoresS, 3);

        dCatMatches = abs(nCatMatches(1,:) - round(supervisorPredictedScores(k, 1)));
        [minDistVotes, DI] = sort(dCatMatches, 2, 'ascend');

        predictedLabelScores = zeros([nModels 1]);
        dMin = minDistVotes(1);
        for l=1:nClasses
            if minDistVotes(l) > dMin
                break;
            end
            predictedLabelScores(l) = mean(predictedScoresS(1, DI(l), predictedLabelsS(1,:) == predictedLabelsCat(DI(l))));
        end
        [predictedLabelScoreSuper, PI] = max(predictedLabelScores, [], 1);
        predictedLabelSuper = predictedLabelsCat(DI(PI));


        correctClass = testSets{i}.Labels(j);
        predictedLabel = predictedLabels{1};

        idxsPredictedCat = find(predictedLabelsCat == categorical(predictedLabels(1)));
        idxsWonModels = predictedLabelsS(1,:) == predictedLabels(1,:);
        predictedLabelScore = mean(predictedScoresS(1, idxsPredictedCat, idxsWonModels));

        fprintf(fd, "%s %f %s %s %f %f %f %d %d %d %f %s", correctClass, predictedLabelScore, predictedLabel, predictedLabelSuper{1},...
            supervisorPredictedScores(k, 1),...
            TrTrain,...
            supervisorPredictedScores(k, 2),...
            supervisorPredictedScores(k, 1) > (nEnsLabels-1)/2.,...
            supervisorPredictedScores(k, 1) > TrTrain,...
            supervisorPredictedScores(k, 1) > supervisorPredictedScores(k, 2),...
            VerdTS(k, 1),...
            testSets{i}.Files{j});
        for l=1:nClasses
            fprintf(fd, " %f", predictedScores(1, l));
        end
        fprintf(fd, "\n");


        % Ask Oracle if in doubt
        difPredLabel = supervisorPredictedScores(k, 2) - supervisorPredictedScores(k, 1);
        if (((mcmcF == 0) && (supervisorPredictedScores(k, 1) < supervisorPredictedScores(k, 2)) && (nOracleAnsw/k < nOrcaleLimit)) ||...
                ((mcmcF == 1) && (difPredLabel > difPredLabelMax)))... 
                

            % Active re-training - insert current test image
            activeFiles = activeSet.Files;
            activeLabels = activeSet.Labels;
            activeFiles(activeCurI) = testSet.Files(j);
            activeSet = imageDatastore(activeFiles);
            activeSet.Labels = activeLabels;
            activeSet.Labels(activeCurI) = testSet.Labels(j);
            activeSet.ReadFcn = trainingSet.ReadFcn;
            for s=1:nModels 
                [myNets(s), TInfo] = trainNetwork(activeSet, myNets(s).layerGraph, aOpts);
            end

            activeCurI = 1 + mod(nOracleAnsw, mb_size);
            %if activeCurI == 0
            %    activeCurI = mb_size;
            %end

            nOracleAnsw = nOracleAnsw + 1;
            difPredLabelMax = difPredLabel;
        end


        % Naive continous learning, retrain every prediction
        if contF
            %sLayers = super2Net.Layers;
            %sgraph = layerGraph(sLayers);
            contActTS(contCurI, :) = ActTS(k, :);
            contVerdTS(contCurI, :) = VerdTS(k, :);
            super2Net = trainNetwork(contActTS, contVerdTS, super2Net.layerGraph, cOptions);

            contCurI = 1 + mod(k, mb_sizeS);
        end
    
    end
    nImgsCur = nImgsCur + nImages;
    
end

%% Sorted activations of model candidates
%[ActTC, IT] = sort(ActT, 2, 'descend');

%StrongT(:, :) = ActTC(:, 1, :);

%% Sort other models by their strongest softmax 
%[StrongTC, IStrongT] = sort(StrongT, 2, 'descend');

% And sort their softmax activations in the same way as activations
% in the model with strongest right softmax
%for k=1:nImgsTot
      
%    for si=1:nModels
        
%        s = IStrongT(k, si);
%        ActTS(k, nClasses*(si-1)+1:nClasses*si) = ActT(k, IT(k, :, IStrongT(k, 1)), s);
        
%    end
%    VerdTS(k,1) = sum(VerdT(k, :), 2);
%    ActTS(k, nClasses*nModels+dimLabel) = VerdTS(k,1);
%end

%[ActTS, VerdTS] = makeUSDstrong(ActT, VerdT, ActTS, VerdTS, nClasses, 0, nImgsTot, nModels);
   
%%         
%sOptions2 = trainingOptions('adam', ...
%        'ExecutionEnvironment','auto',...
%        'Shuffle', 'every-epoch',...
%        'MiniBatchSize', 1, ...
%        'InitialLearnRate',0.01, ...
%        'MaxEpochs',200, ...
%        'Verbose',true);%, ...
        %'Plots','training-progress');

% Supervisor network
%supervisorPredictedScores = zeros([nImgsTot nCLOut]);


nImgsCur = 0;
for i=1:nMakeups 
    clear('predictedScoresS');
    clear('predictedLabelsS');
    
    fprintf('----> Super Makeup # %d/%d\n', i, nMakeups);

    for s=1:nModels 
        predictedScoresS(:, :, s) = predictedScoresSwarm{i, s};
        predictedLabelsS(:, s) = predictedLabelsSwarm{i, s};
    end
    
    % Ensemble voting
    nCatMatches = countcats(predictedLabelsS, 2);
    [MaxCountLabels, MI] = max(nCatMatches, [], 2);
    predictedLabelsCat = (categories(predictedLabelsS));
    predictedLabels = predictedLabelsCat(MI);
    
    predictedScores = mean(predictedScoresS, 3);
        
    [nImages, ~] = size(testSets{i}.Files);
    %%supervisorPredictedScores(1+nImgsCur:nImgsCur+nImages,:) = predict(super2Net, ActTS(1+nImgsCur:nImgsCur+nImages, :));

    for k=1:nImages

        supervisorPredictedScores(nImgsCur+k,:) = predict(super2Net, ActTS(nImgsCur+k, :));

        % find label with number of occurance as predicted by superNet
        dCatMatches = abs(nCatMatches(k,:) - round(supervisorPredictedScores(nImgsCur+k, 1)));
        [minDistVotes, DI] = sort(dCatMatches, 2, 'ascend');

        predictedLabelScores = zeros([nModels 1]);
        dMin = minDistVotes(1);
        for l=1:nClasses
            if minDistVotes(l) > dMin
                break;
            end
            predictedLabelScores(l) = mean(predictedScoresS(k, DI(l), predictedLabelsS(k,:) == predictedLabelsCat(DI(l))));
        end
        [predictedLabelScoreSuper, PI] = max(predictedLabelScores, [], 1);
        predictedLabelSuper = predictedLabelsCat(DI(PI));


        correctClass = testSets{i}.Labels(k);
        predictedLabel = predictedLabels{k};

        idxsPredictedCat = find(predictedLabelsCat == categorical(predictedLabels(k)));
        idxsWonModels = predictedLabelsS(k,:) == predictedLabels(k,:);
        predictedLabelScore = mean(predictedScoresS(k, idxsPredictedCat, idxsWonModels));


        fprintf(fd, "%s %f %s %s %f %f %f %d %d %d %f %s", correctClass, predictedLabelScore, predictedLabel, predictedLabelSuper{1},...
            supervisorPredictedScores(nImgsCur+k, 1),...
            TrTrain,...
            supervisorPredictedScores(nImgsCur+k, 2),...
            supervisorPredictedScores(nImgsCur+k, 1) > (nEnsLabels-1)/2.,...
            supervisorPredictedScores(nImgsCur+k, 1) > TrTrain,...
            supervisorPredictedScores(nImgsCur+k, 1) > supervisorPredictedScores(nImgsCur+k, 2),...
            VerdTS(nImgsCur+k, 1),...
            testSets{i}.Files{k});
        for l=1:nClasses
            fprintf(fd, " %f", predictedScores(k, l));
        end
        fprintf(fd, "\n");

        % Naive continous learning, retrain every prediction
        %sLayers = super2Net.Layers;
        %sgraph = layerGraph(sLayers);
        %super2Net = trainNetwork(ActTS(nImgsCur+k, :), VerdTS(nImgsCur+k, :), sgraph, sOptions2);
        
    end
    
    % Naive continous learning, retrain by grooup in intervals
    %%sLayers = super2Net.Layers;
    %%sgraph = layerGraph(sLayers);
    %%super2Net = trainNetwork(ActTS(1+nImgsCur:nImgsCur+nImages, :), VerdTS(1+nImgsCur:nImgsCur+nImages, :), sgraph, sOptions2);

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
