close all;  % Close all figures
clear;      % Clear workspace variables
clc;        % Clear command window
format compact;

rng(1);  % Set random seed using rng instead of rand('seed',1)

% Load the iris dataset
[x, t] = iris_dataset;
net = patternnet(10);

% Train the neural network
[net, tr] = train(net, x, t);
view(net);

% Evaluate the trained network on the entire dataset
y = net(x);
classes = vec2ind(y);

% Calculate performance on the entire dataset
perf_all = perform(net, t, y);

% Calculate performance on the test set
perf_test = perform(net, t(:, tr.testInd), net(x(:, tr.testInd)));

% Plot confusion matrix for the test set
plotconfusion(t(:, tr.testInd), y(:, tr.testInd));
