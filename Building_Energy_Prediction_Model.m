clear; clc; close all;

%% 1. DATA LOADING & PREPROCESSING
disp('--- Step 1: Loading and Preprocessing Data ---');

% Load training and testing datasets using standard readtable function
trainData = readtable('train.csv');
testData = readtable('test.csv');

% Identify categorical columns and convert them to categorical datatype
catCols = {'State_Factor', 'building_class', 'facility_type'};
for i = 1:length(catCols)
    if ismember(catCols{i}, trainData.Properties.VariableNames)
        trainData.(catCols{i}) = categorical(trainData.(catCols{i}));
    end
    if ismember(catCols{i}, testData.Properties.VariableNames)
        testData.(catCols{i}) = categorical(testData.(catCols{i}));
    end
end

% Handle missing values in numeric columns by imputing with the median
numVars = trainData.Properties.VariableNames(~ismember(trainData.Properties.VariableNames, [catCols, {'site_eui', 'id'}]));
for i = 1:length(numVars)
    colName = numVars{i};
    if any(ismissing(trainData.(colName)))
        trainData.(colName) = fillmissing(trainData.(colName), 'median');
    end
    if ismember(colName, testData.Properties.VariableNames) && any(ismissing(testData.(colName)))
        testData.(colName) = fillmissing(testData.(colName), 'median');
    end
end

% Separate features (X) and target (Y) from training data
X = trainData(:, ~ismember(trainData.Properties.VariableNames, {'site_eui', 'id'}));
Y = trainData.site_eui;

% Partition data into Training (80%) and Testing (20%) sets using cvpartition
cv = cvpartition(height(trainData), 'HoldOut', 0.2);
trainIdx = training(cv);
testIdx = test(cv);

XTrain = X(trainIdx, :);
YTrain = Y(trainIdx);
XTestSet = X(testIdx, :);
YTestSet = Y(testIdx);

fprintf('Data preprocessing complete.\nTraining set size: %d rows\nTesting set size: %d rows\n\n', ...
    height(XTrain), height(XTestSet));

%% 2. MODEL TRAINING (HIGH-ACCURACY BOOSTED TREES)
disp('--- Step 2: Model Training (Gradient Boosting) ---');

% Configure deeper base trees for complex feature interactions
t = templateTree('MaxNumSplits', 60, 'MinLeafSize', 5);

% Train an optimized Gradient Boosted Ensemble (LSBoost) for superior accuracy
% Note: 'Verbose' has been removed to ensure compatibility across MATLAB versions.
model = fitrensemble(XTrain, YTrain, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', 150, ...
    'LearnRate', 0.1, ...
    'Learners', t);

disp('Model training complete.');

%% 3. MODEL EVALUATION & TEST PREDICTION
disp('--- Step 3: Model Evaluation and Test Prediction ---');

% Predict outcomes on the test partition dataset
YPred = predict(model, XTestSet);

% Calculate regression evaluation metrics (RMSE, MAE, R-squared)
rmseVal = sqrt(mean((YTestSet - YPred).^2));
maeVal = mean(abs(YTestSet - YPred));
ssRes = sum((YTestSet - YPred).^2);
ssTot = sum((YTestSet - mean(YTestSet)).^2);
rSquared = 1 - (ssRes / ssTot);

fprintf('\n--- Test Set Performance Metrics ---\n');
fprintf('Root Mean Squared Error (RMSE): %.4f\n', rmseVal);
fprintf('Mean Absolute Error (MAE):      %.4f\n', maeVal);
fprintf('R-Squared (R^2):                %.4f\n\n', rSquared);

% Generate evaluation plots
figure('Name', 'Regression Model Evaluation', 'Position', [100, 100, 1100, 450]);

% Plot 1: Actual vs. Predicted Scatter Plot
subplot(1, 2, 1);
scatter(YTestSet, YPred, 15, 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
lims = [min([YTestSet; YPred]), max([YTestSet; YPred])];
plot(lims, lims, 'r--', 'LineWidth', 1.5);
xlabel('Actual Site EUI');
ylabel('Predicted Site EUI');
title('Actual vs. Predicted Site EUI');
grid on;

% Plot 2: Residual Plot
subplot(1, 2, 2);
residuals = YTestSet - YPred;
scatter(YPred, residuals, 15, 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
yline(0, 'r--', 'LineWidth', 1.5);
xlabel('Predicted Site EUI');
ylabel('Residuals (Actual - Predicted)');
title('Residual Plot');
grid on;

% Predict on the external 'test.csv' dataset
XTestFinal = testData(:, ~ismember(testData.Properties.VariableNames, {'id'}));
testPredictions = predict(model, XTestFinal);

% Save test predictions to a CSV file
if ismember('id', testData.Properties.VariableNames)
    resultsTable = table(testData.id, testPredictions, 'VariableNames', {'id', 'site_eui'});
else
    resultsTable = table((1:height(testData))', testPredictions, 'VariableNames', {'id', 'site_eui'});
end
writetable(resultsTable, 'sample_solution.csv');
disp('Test predictions successfully generated and saved to "test_predictions.csv".');