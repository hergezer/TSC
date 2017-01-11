% Implementation of feature covariance based time series classification in
% Adiac dataset of UCR Repository

% Hamza Ergezer - 2016

clear all;close all;
load('.\UCR_Data\Adiac\Adiac_TRAIN')
load('.\UCR_Data\Adiac\Adiac_TEST')
load('optParamsAdiac')

trainDataAll = Adiac_TRAIN(:,2:end);
classesTrain = Adiac_TRAIN(:,1);

dimension = size(trainDataAll,2);

testDataAll = Adiac_TEST(:,2:end);
classesTEST = Adiac_TEST(:,1);

nofFeatures = 6;

bagSize = round(dimension/ratioBlocks);
bagPoints1 = ceil(1:bagSize*(1-overlapRatio):dimension-bagSize);
bagPoints2 = ceil(bagSize:bagSize*(1-overlapRatio):dimension);

nofWords = length(bagPoints1);
classes = unique(classesTrain);
nofClasses = length(classes);
nofTrainInstances = size(trainDataAll,1);
nofTestInstances = size(testDataAll,1);

C_all_Train = zeros(nofFeatures,nofFeatures,nofTrainInstances,nofWords);
C_all_Test = zeros(nofFeatures,nofFeatures,nofTestInstances,nofWords);
distancesWords = zeros(nofTestInstances,nofTrainInstances,nofWords);
distances = zeros(nofTestInstances,nofTrainInstances);

for j=1:nofWords
    trainData = trainDataAll(:,bagPoints1(j):bagPoints2(j));
    rankTrainData = sort(trainData,2);
    tNorm = (2:size(trainData,2))/size(trainData,2);
    
    maxTrainData = max(trainData,[],2);
    maxTrainData = repmat(maxTrainData,1,size(trainData,2));
    devFromMaxTrain = maxTrainData - trainData;
    
    meanTrainData = mean(trainData,2);
    meanTrainData = repmat(meanTrainData,1,size(trainData,2));
    meanTrainData = meanTrainData./maxTrainData;
    devFromMeanTrain = meanTrainData - trainData;
    
    cumsumTrainData = cumsum(trainData,2);
    diffTrainData = diff(trainData,1,2);
    nofTrainInstances = size(trainData,1);
    for i=1:nofTrainInstances
        f = [trainData(i,1:end-1); rankTrainData(i,1:end-1); diffTrainData(i,:); cumsumTrainData(i,1:end-1);devFromMeanTrain(i,1:end-1); tNorm]';
        C_all_Train(:,:,i,j) = logm( cov(f) + lambda*eye(nofFeatures));
    end

    testData = testDataAll(:,bagPoints1(j):bagPoints2(j));
    tNorm = (2:size(testData,2))/size(testData,2);
    rankTestData = sort(testData,2);
    
    maxTestData = max(testData,[],2);
    maxTestData = repmat(maxTestData,1,size(testData,2));
    devFromMaxTest = maxTestData - testData;
    
    meanTestData = mean(testData,2);
    meanTestData = repmat(meanTestData,1,size(testData,2));
    meanTestData = meanTestData./maxTestData;
    devFromMeanTest = meanTestData - testData;
    
    cumsumTestData = cumsum(testData,2);
    diffTestData = diff(testData,1,2);
    nofTestInstances = size(testData,1);
    for i=1:nofTestInstances
        f = [testData(i,1:end-1); rankTestData(i,1:end-1); diffTestData(i,:); cumsumTestData(i,1:end-1);devFromMeanTest(i,1:end-1); tNorm]';
        C_all_Test(:,:,i,j) = logm( cov(f) + lambda*eye(nofFeatures));
    end
end

resultClassesNN = zeros(nofTestInstances,1);
for i=1:nofTestInstances
    for j=1:nofTrainInstances
        for k=1:nofWords
            distancesWords(i,j,k) = norm(C_all_Train(:,:,j,k)-C_all_Test(:,:,i,k),'fro');
        end
        distances(i,j) = mean(distancesWords(i,j,:));
    end
    [~,minInd] = min(distances(i,:));
    resultClassesNN(i) = classesTrain(minInd);    
end

ClassErrorNN = 1 - sum(resultClassesNN == classesTEST)/nofTestInstances
