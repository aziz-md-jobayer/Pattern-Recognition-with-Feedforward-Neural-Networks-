%% Crab Classification
%% Preparing the Data
% Load the dataset
load crab_dataset;
[x, t] = crab_dataset;
fprintf('Size of x: %s\n', mat2str(size(x)));
fprintf('Size of t: %s\n', mat2str(size(t)));

%% Building the Neural Network Classifier
% Set the random seed for reproducibility
setdemorandstream(491218382);

% Create a neural network with 10 neurons in the hidden layer
net = patternnet(10);
view(net);

% Train the neural network
[net, tr] = train(net, x, t);
nntraintool; % Open the Neural Network Training Tool
plotperform(tr);

%% Testing the Classifier
% Extract test data
testX = x(:, tr.testInd);
testT = t(:, tr.testInd);

% Run the trained network on the test data
testY = net(testX);
testIndices = vec2ind(testY);

%% Displaying Results
% Plot confusion matrix
plotconfusion(testT, testY);

% Calculate overall correct and incorrect classification percentages
[c, cm] = confusion(testT, testY);
fprintf('Percentage Correct Classification: %f%%\n', 100 * (1 - c));
fprintf('Percentage Incorrect Classification: %f%%\n', 100 * c);

% Plot ROC curve
plotroc(testT, testY);

%% Copyright
% Copyright 2012 The MathWorks, Inc.
