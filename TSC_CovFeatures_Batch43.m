clear all;close all;
datasets = getDatasetNamesUCR43;

nofFeatures = 6;
load('optParams')

ClassErrorNN = zeros(length(datasets),1);
timeElapsedFeature = zeros(length(datasets),1);
timeElapsedNN = zeros(length(datasets),1);
for id = 1:length(datasets)
    tic;
    
    savedTrainData = load(['.\UCR_Data\' datasets{id} '\' datasets{id} '_TRAIN']);
    savedTestData = load(['.\UCR_Data\' datasets{id} '\' datasets{id} '_TEST']);
    
    trainDataAll = savedTrainData(:,2:end);
    classesTrain = savedTrainData(:,1);
    
    dimension = size(trainDataAll,2);
    
    testDataAll = savedTestData(:,2:end);
    classesTest = savedTestData(:,1);
    classes = unique(classesTrain);
    nofClasses = length(classes);
    nofTrainInstances = size(trainDataAll,1);
    nofTestInstances = size(testDataAll,1);
    
    ratioBlocks = optParams(id,1);
    overlapRatio = optParams(id,2);    
    
    bagSize = round(dimension/ratioBlocks);
    bagPoints1 = ceil(1:bagSize*(1-overlapRatio):dimension-bagSize);
    bagPoints2 = ceil(bagSize:bagSize*(1-overlapRatio):dimension);
    nofWords = length(bagPoints1);
    lambda = optParams(id,3);
    distances = zeros(nofTestInstances,nofTrainInstances);
        
    C_all_Train = zeros(nofFeatures,nofFeatures,nofTrainInstances,nofWords);
    C_all_Test = zeros(nofFeatures,nofFeatures,nofTestInstances,nofWords);
    
    distancesWords = zeros(nofTestInstances,nofTrainInstances,nofWords);
    for j=1:nofWords
        trainData = trainDataAll(:,bagPoints1(j):bagPoints2(j));
        rankTrainData = sort(trainData,2);
        tNorm = (2:size(trainData,2))/size(trainData,2);
        
        maxTrainData = max(trainData,[],2);
        maxTrainData = repmat(maxTrainData,1,size(trainData,2));
        devFromMaxTrain = maxTrainData - trainData;
        
        meanTrainData = mean(trainData,2);
        meanTrainData = repmat(meanTrainData,1,size(trainData,2));
        meanTrainData = meanTrainData./(maxTrainData+eps);
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
        meanTestData = meanTestData./(maxTestData+eps);
        devFromMeanTest = meanTestData - testData;
        
        cumsumTestData = cumsum(testData,2);
        diffTestData = diff(testData,1,2);
        nofTestInstances = size(testData,1);
        for i=1:nofTestInstances
            f = [testData(i,1:end-1); rankTestData(i,1:end-1); diffTestData(i,:); cumsumTestData(i,1:end-1);devFromMeanTest(i,1:end-1); tNorm]';            
            C_all_Test(:,:,i,j) = logm( cov(f) + lambda*eye(nofFeatures));            
        end
    end

    timeElapsedFeature(id) = toc;

    tic;
    ClassErrorNN(id) = calculateClassificationError(C_all_Train,C_all_Test,classesTrain,classesTest);
    timeElapsedNN(id) = toc;    
end



