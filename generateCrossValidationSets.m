function [trainDataAll, testDataAll, classesTrain, classesTest] = generateCrossValidationSets(DataAll,classesAll)
classes = unique(classesAll);
trainDataAll = [];
testDataAll = [];
classesTrain = [];
classesTest = [];
for i = 1:length(classes)
    class_i = classes(i);
    ind_i = find(classesAll==class_i);
    nof_i = length(ind_i);
    randInd_i = ind_i(randperm(nof_i));
    half_nof_i = floor(nof_i/2);
    trainInds_i = randInd_i(1:half_nof_i);
    testInds_i = randInd_i(half_nof_i+1:end);
    trainDataAll_i = DataAll(trainInds_i,:);
    testDataAll_i = DataAll(testInds_i,:);
    classesTrain_i = classesAll(trainInds_i);
    classesTest_i = classesAll(testInds_i);
    trainDataAll = [trainDataAll; trainDataAll_i];
    testDataAll = [testDataAll; testDataAll_i];
    classesTrain = [classesTrain; classesTrain_i];
    classesTest = [classesTest; classesTest_i];
end
    
    