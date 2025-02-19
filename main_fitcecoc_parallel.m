%% File: main_fitcecoc_parallel.m 
%----------------------------------------------------------------
% This script trains a facial recognition model using Singular Value 
% Decomposition (SVD) for feature extraction and Support Vector Machine 
% (SVM) for classification. It uses parallel processing for both image 
% reading and SVD computation.
%----------------------------------------------------------------

clear; clc; close all;

%% Define Parameters
targetSize = [128, 128];  % Image size for processing
k = 300;                   % Number of eigenfaces (features) to keep
location = fullfile('lfw/');  

%% Load Dataset from "lfw"
disp('Creating image datastore...');
imds = imageDatastore(location, 'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @(filename) imresize(im2gray(imread(filename)), targetSize));

% Count images per person and filter those with 10 to 80 images
tbl = countEachLabel(imds);
mask = tbl{:,2} >= 10 & tbl{:,2} <= 40;
persons = unique(tbl{mask,1});  % Use unique to guarantee ordering
imds = subset(imds, ismember(imds.Labels, persons));

disp(['Using ', num2str(numel(persons)), ' persons for training, each with 10-40 images.']);

%% Display Sample Images
t = tiledlayout(3, 4, 'TileSpacing', 'compact'); % Define layout
nexttile;
montage(imds.Files(1:min(16, numel(imds.Files)))); % Display first 16 images
title('Sample Faces');

%% Read and Prepare Data Matrix (Parallelized)
disp('Reading all images...');
numImages = numel(imds.Files);
% Use single precision
B = zeros([prod(targetSize), numImages], 'single');

% Read images in parallel; each image is flattened and normalized.
parfor i = 1:numImages
    img = imresize(im2gray(imread(imds.Files{i})), targetSize);
    % Convert to single and scale by 1/256
    B(:, i) = single(img(:)) ./ 256;
end


disp('Normalizing data...');
% Normalize the data (this subtracts the mean and divides by standard deviation)
% and returns normalization parameters C and SD.
[B, C, SD] = normalize(B);

%% Perform Global Singular Value Decomposition (SVD) in Parallel
disp('Computing global SVD using distributed arrays...');
tic;

% Convert B into a distributed array for parallel SVD computation.
B_dist = distributed(B);

% Compute the global SVD on the distributed matrix.
[U_dist, S_dist, V_dist] = svd(B_dist, 'econ');

% Gather results back to the client.
U_full = gather(U_dist);
S_full = gather(S_dist);
V_full = gather(V_dist);
toc;

% Extract singular values
singularValues = diag(S_full);

% Total energy (sum of squares of all singular values)
totalEnergy = sum(singularValues.^2);

% Energy in the top k singular values
energyTopK = sum(singularValues(1:k).^2);

% Fraction (or percentage) of energy retained
energyRetained = energyTopK / totalEnergy;
fprintf('Energy retained with top %d eigenfaces: %.2f%%\n', k, energyRetained*100);

% Get eigenfaces (principal components) from the global SVD.
k = min(size(V_full, 2), k);  % Ensure k doesn't exceed available features
% Keep top k eigenfaces. (Note: With B = U * S * V', we have U' * B = S * V', so
% computing the projection via U' is equivalent to S*V'.)
U = U_full(:, 1:k);
% Compute feature weights exactly as in recognize_faces:
W = S_full(1:k, 1:k) * V_full(:, 1:k)';  % This equals U' * B

%% Display Eigenfaces
% Create a cell array of the top 16 eigenfaces for display.
Eigenfaces = arrayfun(@(j) reshape((U(:,j) - min(U(:,j))) ./ (max(U(:,j)) - min(U(:,j))), ...
    targetSize), 1:min(16, size(U,2)), 'UniformOutput', false);

nexttile;
montage(Eigenfaces);  % Show top 16 eigenfaces
title('Top 16 Eigenfaces');
colormap(gray);

%% Prepare Data for Classification
% In training, we compute features as S*V' so that (since B = U*S*V') we have
% U' * B = S*V'. This is identical to what the instructor's recognize_faces does.
X = W';  % Feature matrix (observations as rows)
Y = categorical(imds.Labels, persons);  % Convert labels to categorical

% Assign colors to target values for visualization.
cm = lines(numel(persons));
c = cm(1+mod(uint8(Y), size(cm,1)), :);

%% Train Support Vector Machine (SVM) on Full Dataset
disp('Training Support Vector Machine on full dataset with parallel options...');
options = statset('UseParallel', true);
tic;
Mdl = fitcecoc(X, Y, 'Learners', 'svm', 'Options', options);
toc;

%% Predict on Full Dataset in Parallel
disp('Predicting on full dataset using parfor...');
numObs = size(X,1);
pool = gcp();
numWorkers = pool.NumWorkers;

% Partition the data so that each worker handles roughly equal rows.
base = floor(numObs/numWorkers);
remainder = mod(numObs, numWorkers);
batchSizes = repmat(base, numWorkers, 1);
batchSizes(1:remainder) = batchSizes(1:remainder) + 1;

indices = cell(numWorkers,1);
startIdx = 1;
for i = 1:numWorkers
    endIdx = startIdx + batchSizes(i) - 1;
    indices{i} = startIdx:endIdx;
    startIdx = endIdx + 1;
end

YPred_batches = cell(numWorkers,1);
Score_batches = cell(numWorkers,1);

parfor i = 1:numWorkers
    idx = indices{i};
    [YPred_batches{i}, Score_batches{i}] = predict(Mdl, X(idx,:));
end

% Combine results from each worker.
YPred = vertcat(YPred_batches{:});
Score = vertcat(Score_batches{:});

%% Visualization: Feature Space and Performance Metrics
nexttile;
scatter3(X(:,1), X(:,2), X(:,3), 50, c);
title('Top 3-Predictor Space');
xlabel('x1'); ylabel('x2'); zlabel('x3');

nexttile;
scatter3(X(:,4), X(:,5), X(:,6), 50, c);
title('Next 3-Predictor Space');
xlabel('x4'); ylabel('x5'); zlabel('x6');

% Compute and plot ROC metrics.
disp('Plotting ROC metrics...');
rm = rocmetrics(Y, Score, Mdl.ClassNames);
nexttile;
plot(rm);

disp('Plotting confusion matrix...');
nexttile;
confusionchart(Y, YPred);
title(['Number of features: ' , num2str(k)]);

%% Save Trained Model (including normalization parameters)
disp('Saving trained model...');
% Save all variables that are used by the instructor's recognize_faces function.
save('model', 'Mdl', 'persons', 'U', 'targetSize', 'C', 'SD');
disp('Model training complete and saved as "model.mat".');
