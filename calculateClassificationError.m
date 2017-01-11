function classError = calculateClassificationError(C_all_Train,C_all_Test,classesTrain,classesTest)
nofTrainInstances = size(C_all_Train,3);
nofTestInstances = size(C_all_Test,3);
nofWords = size(C_all_Test,4);
distances = zeros(nofTestInstances,nofTrainInstances);
distancesWords = zeros(nofTestInstances,nofTrainInstances,nofWords);
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
classError = 1 - sum(resultClassesNN == classesTest)/nofTestInstances;