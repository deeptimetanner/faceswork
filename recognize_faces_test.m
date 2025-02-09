%----------------------------------------------------------------
% File:     recognize_faces_test.m
%----------------------------------------------------------------
%
% Author:   Marek Rychlik (rychlik@arizona.edu)
% Date:     Sun Dec  8 15:55:17 2024
% Copying:  (C) Marek Rychlik, 2020. All rights reserved.
% 
%----------------------------------------------------------------
% A driver for function RECOGNIZE_FACES.
%
% It passes a number of images from the database of images
% and predicts the labels.
%
% As a visualization tool, it burns predicted labels into the images
% and displays a montage.

location = fullfile('lfw/');

disp('Creating image datastore...');
% We read images in their original format0
imds0 = imageDatastore(location,...
                       'IncludeSubfolders',true,...
                       'LabelSource','foldernames');

load('model',["persons"])

idx = ismember(imds0.Labels, persons);
imds = subset(imds0, idx);

RGB = readall(imds);

Y = imds.Labels;

YPred = recognize_faces(RGB);

% Burn labels into the images
for j=1:numel(RGB)
    RGBannotated{j} = insertObjectAnnotation(RGB{j}, 'rectangle', [10,10,100,20], YPred(j));
end

montage(RGBannotated);

Accuracy = numel(find(Y==YPred))/numel(Y);
disp(['Percentage of correctly labeled images: ', num2str(100*Accuracy),'%']);