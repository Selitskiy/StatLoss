function [ActS, VerdS] = makeUSDstrong(Act, Verd, ActS, VerdS, nClasses, k, nImgs, nModels, dimLabel)

    [ActC, IT] = sort(Act, 2, 'descend');

    Strong(:, :) = ActC(:, 1, :);

    %% Sort other models by their strongest softmax 
    [StrongC, IStrong] = sort(Strong, 2, 'descend');

    % And sort their softmax activations in the same way as activations
    % in the model with strongest right softmax
    if k == 0
        for k=1:nImgs
      
            for si=1:nModels
        
                s = IStrong(k, si);
                ActS(k, nClasses*(si-1)+1:nClasses*si) = Act(k, IT(k, :, IStrong(k, 1)), s);
        
            end
            VerdS(k,1) = sum(Verd(k, :), 2);
            ActS(k, nClasses*nModels+dimLabel) = VerdS(k,1);
        end
    else
        for si=1:nModels
        
            s = IStrong(k, si);
            ActS(k, nClasses*(si-1)+1:nClasses*si) = Act(k, IT(k, :, IStrong(k, 1)), s);
        
        end
        VerdS(k,1) = sum(Verd(k, :), 2);
        ActS(k, nClasses*nModels+dimLabel) = VerdS(k,1);
    end

end