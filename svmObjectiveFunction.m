% In svmObjectiveFunction.m
function loss = svmObjectiveFunction(params, X, Y)
    tempMdl = fitcecoc(X, Y, 'Learners', templateSVM( ...
        'BoxConstraint', params.BoxConstraint, ...
        'KernelScale', params.KernelScale), 'Options', statset('UseParallel', true));
    cvMdl = crossval(tempMdl, 'KFold', 5);
    loss = kfoldLoss(cvMdl);
end
