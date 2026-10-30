clear; clc; close all;

trainData = readtable('train.csv');
testData = readtable('test.csv');

trainData.source = repmat({'train'}, height(trainData), 1);
testData.source = repmat({'test'}, height(testData), 1);
testData.site_eui = nan(height(testData), 1);
df = [trainData; testData];

catCols = {'State_Factor', 'building_class', 'facility_type'};
for i = 1:length(catCols)
    df.(catCols{i}) = categorical(df.(catCols{i}));
end

sortCols = {'State_Factor', 'building_class', 'facility_type', 'floor_area', 'year_built', 'Year_Factor'};
df = sortrows(df, sortCols);

minTempCols = df.Properties.VariableNames(contains(df.Properties.VariableNames, 'min_temp'));
maxTempCols = df.Properties.VariableNames(contains(df.Properties.VariableNames, 'max_temp'));
avgTempCols = df.Properties.VariableNames(contains(df.Properties.VariableNames, 'avg_temp'));

df.min_temp1 = min(df{:, minTempTempColIndices(minTempCols, df)}, [], 2);
df.max_temp1 = min(df{:, maxTempCols}, [], 2);
df.avg_temp1 = min(df{:, avgTempCols}, [], 2);

df.min_temp2 = max(df{:, minTempCols}, [], 2);
df.max_temp2 = max(df{:, maxTempCols}, [], 2);
df.avg_temp2 = max(df{:, avgTempCols}, [], 2);

df.min_temp3 = mean(df{:, minTempCols}, 2);
df.max_temp3 = mean(df{:, maxTempCols}, 2);
df.avg_temp3 = mean(df{:, avgTempCols}, 2);

df.min_temp4 = std(df{:, minTempCols}, 0, 2);
df.max_temp4 = std(df{:, maxTempCols}, 0, 2);
df.avg_temp4 = std(df{:, avgTempCols}, 0, 2);

df.min_temp5 = skewness(df{:, minTempCols}, 0, 2);
df.max_temp5 = skewness(df{:, maxTempCols}, 0, 2);
df.avg_temp5 = skewness(df{:, avgTempCols}, 0, 2);

groupKeys = {'State_Factor', 'building_class', 'facility_type', 'floor_area', 'year_built'};

for i = 1:3
    lagNameEUI = sprintf('site_eui_lag%d', i);
    lagNameYear = sprintf('Year_Factor_lag%d', i);
    lagNameEnergy = sprintf('energy_star_rating_lag%d', i);
    lagNameElev = sprintf('ELEVATION_lag%d', i);
   
    df.(lagNameEUI) = groupedLag(df, groupKeys, 'site_eui', i);
    df.(lagNameYear) = groupedLag(df, groupKeys, 'Year_Factor', i);
    df.(lagNameEnergy) = groupedLag(df, groupKeys, 'energy_star_rating', i);
    df.(lagNameElev) = groupedLag(df, groupKeys, 'ELEVATION', i);
   
    elevFloorName = sprintf('ELEVATION_FLOOR_lag%d', i);
    elevFloorBName = sprintf('ELEVATION_FLOORB_lag%d', i);
    df.(elevFloorName) = df.(lagNameElev) .* df.floor_area;
    df.(elevFloorBName) = df.(lagNameElev) ./ df.floor_area;
   
    deltaEnergyName = sprintf('delta_energy%d', i);
    deltaEnergyBName = sprintf('delta_energy%d_B', i);
    deltaEUIBName = sprintf('site_eui%dB', i);
    deltaEUICName = sprintf('site_eui%dC', i);
   
    validEnergy = ~isnan(df.energy_star_rating) & ~isnan(df.(lagNameEnergy));
    df.(deltaEnergyName) = nan(height(df), 1);
    df.(deltaEnergyName)(validEnergy) = (df.energy_star_rating(validEnergy) - df.(lagNameEnergy)(validEnergy)) ./ df.energy_star_rating(validEnergy);
   
    df.(deltaEnergyBName) = nan(height(df), 1);
    df.(deltaEnergyBName)(validEnergy) = df.energy_star_rating(validEnergy) - df.(lagNameEnergy)(validEnergy);
   
    validEUIB = ~isnan(df.(lagNameEUI)) & ~isnan(df.(deltaEnergyBName));
    df.(deltaEUIBName) = nan(height(df), 1);
    df.(deltaEUIBName)(validEUIB) = df.(lagNameEUI)(validEUIB) .* df.(deltaEnergyBName)(validEUIB);
   
    validEUIC = ~isnan(df.(lagNameEUI)) & ~isnan(df.(deltaEUIBName));
    df.(deltaEUICName) = nan(height(df), 1);
    df.(deltaEUICName)(validEUIC) = df.(lagNameEUI)(validEUIC) - df.(deltaEUIBName)(validEUIC);
end

df.ELEVATION_FLOOR = df.ELEVATION .* df.floor_area;
df.ELEVATION_FLOORB = df.ELEVATION ./ df.floor_area;

numVars = df.Properties.VariableNames(~ismember(df.Properties.VariableNames, [catCols, {'site_eui', 'id', 'source'}]));
for i = 1:length(numVars)
    col = numVars{i};
    if any(ismissing(df.(col)))
        df.(col) = fillmissing(df.(col), 'median');
    end
end

X = df(strcmp(df.source, 'train'), ~ismember(df.Properties.VariableNames, {'site_eui', 'id', 'source'}));
Y = df.site_eui(strcmp(df.source, 'train'));

t = templateTree('MaxNumSplits', 80, 'MinLeafSize', 5, 'Surrogate', 'on');
model = fitrensemble(X, Y, 'Method', 'LSBoost', 'NumLearningCycles', 200, 'LearnRate', 0.05, 'Learners', t);

XTestFinal = df(strcmp(df.source, 'test'), ~ismember(df.Properties.VariableNames, {'site_eui', 'id', 'source'}));
testPredictions = predict(model, XTestFinal);

testSubset = df(strcmp(df.source, 'test'), :);
if ismember('id', testSubset.Properties.VariableNames)
    resultsTable = table(testSubset.id, testPredictions, 'VariableNames', {'id', 'site_eui'});
else
    resultsTable = table((1:height(testSubset))', testPredictions, 'VariableNames', {'id', 'site_eui'});
end

writetable(resultsTable, 'test_predictions.csv');

function lagVals = groupedLag(T, groupCols, targetCol, n)
    [G, ID] = findgroups(T(:, groupCols));
    target = T.(targetCol);
    lagVals = nan(height(T), 1);
    for k = 1:max(G)
        idx = (G == k);
        vals = target(idx);
        if length(vals) > n
            lagVals(idx) = [nan(n, 1); vals(1:end-n)];
        end
    end
end

function idxs = minTempTempColIndices(names, T)
    idxs = false(1, width(T));
    for i = 1:length(names)
        idxs(strcmp(T.Properties.VariableNames, names{i})) = true;
    end
end