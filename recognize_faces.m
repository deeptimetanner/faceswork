function YPred = recognize_faces(RGB)
% RECOGNIZE_FACES maps images of faces to people names.
%   YPred = recognize_faces(RGB) accepts a cell array of RGB images and 
%   returns a categorical array of predicted labels.
%
% IMPORTANT: This function uses the same preprocessing steps as used during 
% training by applying the training normalization parameters.
    
    % Load precomputed model and normalization parameters.
    load('model.mat', 'Mdl', 'persons', 'U', 'targetSize', 'C', 'SD');
    
    num_images = numel(RGB);
    
    % Convert each image to grayscale and resize to targetSize.
    Grayscale = cellfun(@(I) imresize(im2gray(I), targetSize), RGB, 'UniformOutput', false);
    
    % Flatten images into columns of a matrix.
    D = prod(targetSize);
    B = zeros(D, num_images, 'single');
    for i = 1:num_images
        B(:, i) = single(Grayscale{i}(:)) ./ 256;  % Scale by 1/256, as in training.
    end
    
    % Normalize the data using the training parameters.
    B = (B - C) ./ SD;
    
    % Extract feature weights using the precomputed eigenfaces.
    W = U' * B;
    
    % Prepare the feature matrix (observations as rows) and predict.
    X = W';
    YPred = predict(Mdl, X);
end