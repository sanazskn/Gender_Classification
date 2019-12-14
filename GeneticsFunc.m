function [maxFitness,pop]=GeneticsFunc(featureVectors,classLabel,...
kfoldNum,dataIdx,maxIter,popNum,elitismPerc,crossoverTH,...
mutationTH,desiredFeatureNum)

    elitismNum=elitismPerc*popNum;
    featuresNum=length(featureVectors(1,:));
    
    
    oneTrainDataNum=round(((kfoldNum-1)*length(dataIdx(1,:)))/kfoldNum);
    oneCvDataNum=length(dataIdx(1,:))-oneTrainDataNum;
    totalTrainNum=2*oneTrainDataNum;
    totalCvNum=2*oneCvDataNum;
    
    numParentPairs=(popNum-elitismNum)/2;
    pop=true(popNum,desiredFeatureNum);
    pop=[pop,false(popNum,featuresNum-desiredFeatureNum)];
    
    for i=1:1000
        r1=randi([1,featuresNum],popNum,1);
        r2=randi([1,featuresNum],popNum,1);
        for j=1:popNum
             temp=pop(j,r1(j));
             pop(j,r1(j))=pop(j,r2(j));
             pop(j,r2(j))=temp;
        end
    end
    
    for i=1:maxIter
        fitnessVal=zeros(1,popNum);
        for jj=1:kfoldNum
            cvIdx=dataIdx(:,1+(jj-1)*oneCvDataNum:jj*oneCvDataNum);
            trainIdx=[dataIdx(:,1:(jj-1)*oneCvDataNum),dataIdx(:,jj*oneCvDataNum+1:end)];
            
            cvIdx=[cvIdx(1,:),cvIdx(2,:)];
            trainIdx=[trainIdx(1,:),trainIdx(2,:)];
            
            
            for j=1:popNum
                trainFeatureVec=featureVectors(trainIdx,pop(j,:));
                testFeatureVec=featureVectors(cvIdx,pop(j,:));
                cvClassLabel=knnclassify(testFeatureVec,trainFeatureVec,classLabel(trainIdx),1,'euclidean');
                fitnessVal(j)=fitnessVal(j)+sum(cvClassLabel'==classLabel(cvIdx))/totalCvNum;            
            end
        end
        fitnessVal=fitnessVal/kfoldNum;
        [sortedFitness,sortedFitnessIdx]=sort(fitnessVal,'descend');
        maxFitness.fitnessVal(i)=sortedFitness(1);
        maxFitness.fitnessIdx(i)=sortedFitnessIdx(1);
        maxFitness.pop(i,:)=pop(sortedFitnessIdx(1),:);
        if(maxFitness.fitnessVal(i)==1)
            break;
        end
        newPop=logical([]);
        newPop(1:elitismNum,:)=pop(sortedFitnessIdx(1:elitismNum),:);
        probFitness=fitnessVal/sum(fitnessVal);
        probFitness=[0,probFitness];
        cumsumProbFitness=cumsum(probFitness);
        
        for k=1:numParentPairs
            %selection
            randVal1=rand(1,1);
            randVal2=rand(1,1);
            for m=1:length(cumsumProbFitness)
                if(randVal1<=cumsumProbFitness(m))
                    pIdx1=m-1;
                    break;
                end
            end
            for m=1:length(cumsumProbFitness)
                if(randVal2<=cumsumProbFitness(m))
                    pIdx2=m-1;
                    break;
                end
            end
            %crossover
            crossoverRandVal=rand(1,1);
            if(crossoverRandVal<=crossoverTH)
                randCrossoverIdx=randi([2,featuresNum-1],1,1);
                child1=[pop(pIdx1,1:randCrossoverIdx),pop(pIdx2,randCrossoverIdx+1:end)];
                child2=[pop(pIdx2,1:randCrossoverIdx),pop(pIdx1,randCrossoverIdx+1:end)];
            else
                child1=pop(pIdx1,:);
                child2=pop(pIdx2,:);
            end
            % number of features condition
            %mutation
            ssch1=sum(child1);
            if(ssch1>desiredFeatureNum)
                nn=ssch1-desiredFeatureNum;
                [fval,fidx]=find(child1==1);
                ffidx=randperm(length(fidx));
                child1(fidx(ffidx(1:nn)))=0;
            elseif(ssch1<desiredFeatureNum)
                nn=desiredFeatureNum-ssch1;
                [fval,fidx]=find(child1==0);
                ffidx=randperm(length(fidx));
                child1(fidx(ffidx(1:nn)))=1;
            end
            ssch2=sum(child2);
            if(ssch2>desiredFeatureNum)
                nn=ssch2-desiredFeatureNum;
                [fval,fidx]=find(child2==1);
                ffidx=randperm(length(fidx));
                child2(fidx(ffidx(1:nn)))=0;
            elseif(ssch2<desiredFeatureNum)
                nn=desiredFeatureNum-ssch2;
                [fval,fidx]=find(child2==0);
                ffidx=randperm(length(fidx));
                child2(fidx(ffidx(1:nn)))=1;
            end
            
%             mutationRandVal=rand(1,featuresNum);
%             mutationIdx=mutationRandVal<=mutationTH;
%             child1(mutationIdx)=1-child1(mutationIdx);
%             
%             mutationRandVal=rand(1,featuresNum);
%             mutationIdx=mutationRandVal<=mutationTH;
%             child2(mutationIdx)=1-child2(mutationIdx);
            
            
            %
            newPop=[newPop ;child1 ;child2];
        end
        pop=newPop;
    end
end