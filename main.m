clc
clear all
close all
%%
numMale=45;
numFemale=45;
kfoldNum=5;
desiredFeatureNum=120;
knnNum=3;
maxIter=100;
popNum=40;
elitismPerc=0.1;
crossOverTH=0.85;
mutationTH=0.05;
%%
femaleRootPath='f1/';
maleRootPath='m1/';
cnt=1;
for i=1:numFemale
    dr=dir([femaleRootPath num2str(i) '/' 'chin*']);
    dr=dr.name;
    chinImg=imread([femaleRootPath num2str(i) '/' dr]);
    chinLBPHist=LBPFunc(chinImg);
    dr=dir([femaleRootPath num2str(i) '/' 'eye-left*']);
    dr=dr.name;
    eyeImg=imread([femaleRootPath num2str(i) '/' dr]);
    eyeLBPHist=LBPFunc(eyeImg);
    dr=dir([femaleRootPath num2str(i) '/' 'nose*']);
    dr=dr.name;
    noseImg=imread([femaleRootPath num2str(i) '/' dr]);
    noseLBPHist=LBPFunc(noseImg);
    dr=dir([femaleRootPath num2str(i) '/' 'lip*']);
    dr=dr.name;
    mouthImg=imread([femaleRootPath num2str(i) '/' dr]);
    mouthLBPHist=LBPFunc(mouthImg);
    dr=dir([femaleRootPath num2str(i) '/' 'eyebrow-left*']);
    dr=dr.name;
    eyebrowImg=imread([femaleRootPath num2str(i) '/' dr]);
    eyebrowLBPHist=LBPFunc(eyebrowImg);
    
    %featureVectors(cnt,:)=[chinLBPHist,eyeLBPHist,noseLBPHist,mouthLBPHist,eyebrowLBPHist];
    featureVectors(cnt,:)=[chinLBPHist,eyeLBPHist,noseLBPHist,mouthLBPHist];
    classLabel(cnt)=0;
    cnt=cnt+1;
end
for i=1:numMale
    dr=dir([maleRootPath num2str(i) '/' 'chin*']);
    dr=dr.name;
    chinImg=imread([maleRootPath num2str(i) '/' dr]);
    chinLBPHist=LBPFunc(chinImg);
    
    dr=dir([maleRootPath num2str(i) '/' 'eye-left*']);
    dr=dr.name;
    eyeImg=imread([maleRootPath num2str(i) '/' dr]);
    eyeLBPHist=LBPFunc(eyeImg);
    
    dr=dir([maleRootPath num2str(i) '/' 'nose*']);
    dr=dr.name;
    noseImg=imread([maleRootPath num2str(i) '/' dr]);
    noseLBPHist=LBPFunc(noseImg);
    
    dr=dir([maleRootPath num2str(i) '/' 'lip*']);
    dr=dr.name;
    mouthImg=imread([maleRootPath num2str(i) '/' dr]);
    mouthLBPHist=LBPFunc(mouthImg);
    
    dr=dir([maleRootPath num2str(i) '/' 'eyebrow-left*']);
    dr=dr.name;
    eyebrowImg=imread([maleRootPath num2str(i) '/' dr]);
    eyebrowLBPHist=LBPFunc(eyebrowImg);
    
   % featureVectors(cnt,:)=[chinLBPHist,eyeLBPHist,noseLBPHist,mouthLBPHist,eyebrowLBPHist];
    featureVectors(cnt,:)=[chinLBPHist,eyeLBPHist,noseLBPHist,mouthLBPHist];
    classLabel(cnt)=1;
    cnt=cnt+1;
end
dataIdx=[11:45;56:90];% train and cross validation data idx
testDataIdx=[1:10,46:55];

[maxFitness,pop]=GeneticsFunc(featureVectors,classLabel,kfoldNum,dataIdx,...
    maxIter,popNum,elitismPerc,crossOverTH,mutationTH,desiredFeatureNum);
figure;
plot(maxFitness.fitnessVal,'*-r')
figure;
bar(maxFitness.pop(end,:))

%%
oneTrainDataNum=round(((kfoldNum-1)*length(dataIdx(1,:)))/kfoldNum);
totalTrainNum=2*oneTrainDataNum;
oneCvDataNum=length(dataIdx(1,:))-oneTrainDataNum;

 
trainIdx=dataIdx(:);

trainFeatureVec=featureVectors(trainIdx,maxFitness.pop(end,:));    
testFeatureVec=featureVectors(testDataIdx,maxFitness.pop(end,:)); 
testDataNum=length(testDataIdx);
%% KNN Classifier
testClassLabel=knnclassify(testFeatureVec,trainFeatureVec,classLabel(trainIdx),knnNum,'euclidean');    
fitnessVal=100*sum(testClassLabel'==classLabel(testDataIdx))/testDataNum;            
disp(['Classification Rate for KNN:',num2str(sum(fitnessVal)) '%']);
%% SVM Classifier
svmStruct = svmtrain(trainFeatureVec,classLabel(trainIdx));
testClassLabel = svmclassify(svmStruct,testFeatureVec);
fitnessVal=100*sum(testClassLabel'==classLabel(testDataIdx))/testDataNum;
disp(['Classification Rate for SVM:',num2str(sum(fitnessVal)) '%']);
 