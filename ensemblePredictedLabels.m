function [predictedLabels, predictedScores, predictedLabelsCat, predictedLabelSuper, predictedLabelScoreSuper] = ensemblePredictedLabels(...
    predictedLabelsRSw, predictedScoresRSw, supervisorPredictedScores, nImagesR, nClasses)

        nCatMatches = countcats(predictedLabelsRSw, 2); %number of label predictions by ensamble for each class (by image)
        [MaxCountLabels, MI] = max(nCatMatches, [], 2); %number of lables of the highest occuring class (index of that class)
        predictedLabelsCat = (categories(predictedLabelsRSw)); %name of the highest occuring class (with index MI)
        predictedLabels = predictedLabelsCat(MI); %mapping of numeric index MI to class names
    
        predictedScores = mean(predictedScoresRSw, 3); %average scores for classes over ensemble

        %dCatMatches = abs(nCatMatches - round(supervisorPredictedScores(:, 1))); 
        dCatMatches = abs(nCatMatches - supervisorPredictedScores(:, 1)); %distance between number of ensemble predictions for each class 
        % and snn's expected number of correct predictions
        
        [minDistVotes, DI] = sort(dCatMatches, 2, 'ascend'); %sorted distances to snn's prediction for each class, 
        % and indexs of classes with shortest distance (DI(:,1))

        [maxUncertDist, UI] = sort(minDistVotes(:,1), 1, 'descend');
        kMax = UI(1); % image with maximal uncertainty (largest distance to snn's prediction for the best class)

        predictedLabelScores = zeros([nImagesR nClasses]);
        dMin = minDistVotes(:,1); %minimal distances among class
        iMin = DI(:,1); %classes with minimal distance
        for k=1:nImagesR
            for l=1:nClasses
                if sum(predictedLabelsRSw(k,:) == predictedLabelsCat(DI(k,l)))
                    predictedLabelScores(k,DI(k,l)) = mean(predictedScoresRSw(k, DI(k,l), predictedLabelsRSw(k,:) == predictedLabelsCat(DI(k,l))));
                    %average score of classes with minimal distance over ensemble
                end
            end
        end
        [predictedLabelScoreSuper, PI] = max(predictedLabelScores, [], 2);
        predictedLabelSuper = predictedLabelsCat(PI);
end