% Written by Mang Shao and Tae-Kyun Kim, June 20, 2014.
% Updated by T-K Kim, Feb 07, 2016
%
% The codes are made for educational purpose only.
% 
% Some important functions:
%     
% internal functions:
% 
%     getData.m  - Generate training and testing data
% 
% external functions and libraries:
%     
%     VLFeat    - A large open source library implements popular computer vision algorithms. BSD License.
%                 http://www.vlfeat.org/index.html
%     
%     LIBSVM    - A library for support vector machines. BSD License
%                 http://www.csie.ntu.edu.tw/~cjlin/libsvm/
% 
%     mgd.m     - Generates a Multivariate Gaussian Distribution. BSD License
%                 Written by Timothy Felty
%                 http://www.mathworks.co.uk/matlabcentral/fileexchange/5984-multivariate-gaussian-distribution
% 
%     subaxis.m - Modified 'subplot' function. No BSD License
%     parseArgs.m   Written by Aslak Grinsted
%                 http://www.mathworks.co.uk/matlabcentral/fileexchange/3696-subaxis-subplot
% 
%     suptitle.m- Create a "master title" at the top of a figure with several subplots
%                 Written by Drea Thomas
% 
%     Caltech_101 image categorisation dataset
%                 L. Fei-Fei, R. Fergus and P. Perona. Learning generative visual models
%                 from few training examples: an incremental Bayesian approach tested on
%                 101 object categories. IEEE. CVPR 2004, Workshop on Generative-Model
%                 Based Vision. 2004
%                 http://www.vision.caltech.edu/Image_Datasets/Caltech101/
% ---------------------------------------------------------------------------
% 
% Under BSD Licence


% Initialisation
init;


%%
%{
%'-t 2 -c 1 -g 5'

% Select and load dataset
[data_train, data_test] = getData('Toy_Spiral'); % {'Toy_Gaussian', 'Toy_Spiral', 'Toy_Circle', 'Caltech'}


%%%%%%%%%%%%%
% check the training data
    % data_train(:,1:2) : [num_data x dim] Training 2D vectors
    % data_train(:,3) : [num_data x 1] Label of training data, {1,2,3}
    
plot_toydata(data_train);

%%%%%%%%%%%%%
% check the testing data
    % data_test(:,1:2) : [num_data x dim] Testing 2D vectors, 2D points in the
    % uniform dense grid within the range of [-1.5, 1.5]
    % data_train(:,3) : N/A

%%%%%%%%%%%%%%
% one vs all algorithm
scatter(data_test(:,1),data_test(:,2),'.b');
score=[];
class=3;
for i=1:class
    LabelMat=2*double(data_train(:,3)==i)-1;
    SVMModel = svmtrain(LabelMat,data_train(:,[1,2]),'-t 1 -d 2');
    [predict_label, accuracy, prob_estimates]=svmpredict(data_test(:,3),data_test(:,[1,2]),SVMModel);
    score=[score,prob_estimates];
end
[~,result]=max(score,[],2);
data_test(:,3)=result;

figure;

plot_toydata(data_test);
plot_toydata(data_train);

%%%%%%%%%%%%%%%%%%
% one vs one
voteMat=[]
for i=1:class-1
    for j=i+1:class
        data_train_1v1=data_train(data_train(:,3)==i | data_train(:,3)==j,:);
        LabelMat1v1=2*double(data_train_1v1(:,3)==i)-1;
        SVMModel1v1 = svmtrain(LabelMat1v1,data_train_1v1(:,[1,2]),'-t 1 -d 2');
        [predict_label_1v1, accuracy1v1, prob_estimates_1v1]=svmpredict(data_test(:,3),data_test(:,[1,2]),SVMModel1v1);
        vote=(predict_label_1v1+1)/2*i+(-predict_label_1v1+1)/2*j;
        voteMat=[voteMat,vote];
    end
end
result1v1=mode(voteMat,2);
data_test(:,3)=result1v1;
figure;
plot_toydata(data_test);
plot_toydata(data_train);

%%
score=[];
class=3;
for i=1:class
    LabelMat=2*double(data_train(:,3)==i)-1;
    SVMModel = svmtrain(LabelMat,data_train(:,[1,2]),'-t 2 -g 10 -c 10');
    [predict_label, accuracy, prob_estimates]=svmpredict(data_test(:,3),data_test(:,[1,2]),SVMModel);
    score=[score,prob_estimates];
end
[~,result]=max(score,[],2);
data_test(:,3)=result;

figure;
figure('Position', [100, 100, 800, 300]);
subplot(1,2,1);
plot_toydata(data_test);
plot_toydata(data_train);
title('One v.s All, rbf kernel, c=10, gamma=10')

%%%%%%%%%%%%%%%%%%
% one vs one
voteMat=[]
for i=1:class-1
    for j=i+1:class
        data_train_1v1=data_train(data_train(:,3)==i | data_train(:,3)==j,:);
        LabelMat1v1=2*double(data_train_1v1(:,3)==i)-1;
        SVMModel1v1 = svmtrain(LabelMat1v1,data_train_1v1(:,[1,2]),'-t 2 -g 10 -c 10');
        [predict_label_1v1, accuracy1v1, prob_estimates_1v1]=svmpredict(data_test(:,3),data_test(:,[1,2]),SVMModel1v1);
        vote=(predict_label_1v1+1)/2*i+(-predict_label_1v1+1)/2*j;
        voteMat=[voteMat,vote];
    end
end
result1v1=mode(voteMat,2);
data_test(:,3)=result1v1;

subplot(1,2,2);
plot_toydata(data_test);
plot_toydata(data_train);
title('One v.s one, rbf kernel, c=10, gamma=10')
%%
score=[];
class=3;
for i=1:class
    LabelMat=2*double(data_train(:,3)==i)-1;
    SVMModel = svmtrain(LabelMat,data_train(:,[1,2]),'-t 2 -g 100 -c 10');
    [predict_label, accuracy, prob_estimates]=svmpredict(data_test(:,3),data_test(:,[1,2]),SVMModel);
    score=[score,prob_estimates];
end
[~,result]=max(score,[],2);
data_test(:,3)=result;

figure;
figure('Position', [100, 100, 800, 300]);
subplot(1,2,1);
plot_toydata(data_test);
plot_toydata(data_train);
title('One v.s All, rbf kernel, c=10, gamma=100')

%%%%%%%%%%%%%%%%%%
% one vs one
voteMat=[]
for i=1:class-1
    for j=i+1:class
        data_train_1v1=data_train(data_train(:,3)==i | data_train(:,3)==j,:);
        LabelMat1v1=2*double(data_train_1v1(:,3)==i)-1;
        SVMModel1v1 = svmtrain(LabelMat1v1,data_train_1v1(:,[1,2]),'-t 2 -g 100 -c 10');
        [predict_label_1v1, accuracy1v1, prob_estimates_1v1]=svmpredict(data_test(:,3),data_test(:,[1,2]),SVMModel1v1);
        vote=(predict_label_1v1+1)/2*i+(-predict_label_1v1+1)/2*j;
        voteMat=[voteMat,vote];
    end
end
result1v1=mode(voteMat,2);
data_test(:,3)=result1v1;

subplot(1,2,2);
plot_toydata(data_test);
plot_toydata(data_train);
title('One v.s one, rbf kernel, c=10, gamma=100')

%%
score=[];
class=3;
for i=1:class
    LabelMat=2*double(data_train(:,3)==i)-1;
    SVMModel = svmtrain(LabelMat,data_train(:,[1,2]),'-t 2 -g 100 -c 10');
    [predict_label, accuracy, prob_estimates]=svmpredict(data_test(:,3),data_test(:,[1,2]),SVMModel);
    score=[score,prob_estimates];
end
[~,result]=max(score,[],2);
data_test(:,3)=result;

figure;
figure('Position', [100, 100, 800, 300]);
subplot(1,2,1);
plot_toydata(data_test);
plot_toydata(data_train);
title('One v.s All, rbf kernel, c=10, gamma=100');

%%%%%%%%%%%%%%%%%%
% one vs one
voteMat=[]
for i=1:class-1
    for j=i+1:class
        data_train_1v1=data_train(data_train(:,3)==i | data_train(:,3)==j,:);
        LabelMat1v1=2*double(data_train_1v1(:,3)==i)-1;
        SVMModel1v1 = svmtrain(LabelMat1v1,data_train_1v1(:,[1,2]),'-t 2 -g 100 -c 10');
        [predict_label_1v1, accuracy1v1, prob_estimates_1v1]=svmpredict(data_test(:,3),data_test(:,[1,2]),SVMModel1v1);
        vote=(predict_label_1v1+1)/2*i+(-predict_label_1v1+1)/2*j;
        voteMat=[voteMat,vote];
    end
end
result1v1=mode(voteMat,2);
data_test(:,3)=result1v1;

subplot(1,2,2);
plot_toydata(data_test);
plot_toydata(data_train);
title('One v.s one, rbf kernel, c=10, gamma=100')
%}
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for the Caltech101 dataset for image categorisation

% Select and load dataset
%
% Caltech101 dataset: we use 10 classes, 15 images per class, randomly selected, for training, 
    % and 15 other images per class, for testing. 
%
% Feature descriptors: they are multi-scaled dense SIFT features, and their dimension is 128 
    % (for details of the descriptor, if needed, see http://www.vlfeat.org/matlab/vl_phow.html). 
    % We randomly select 100k descriptors for K-means clustering for building the visual vocabulary (due to memory issue).
%
% data_train: [num_data x (dim+1)] Training vectors with class labels
% data_train(:,end): class labels
% data_query: [num_data x (dim+1)] Testing vectors with class labels
% data_query(:,end): class labels

% Set 'showImg = 0' in getData.m if you want to stop displaying training and testing images. 
% Complete getData.m by writing your own lines of code to obtain the visual vocabulary and the bag-of-words histograms for both training and testing data. 


%[data_train, data_test] = getData('Caltech');
%save data_train;
%save data_test;
%load data_train;
load data_test;


%%
%Set 1
%Testing Accuracy

%This code shows an exhausting search over parameters for svm training
%Both one v one and one v all are included

%Variable Initialisation
Accuracy1VAMat=[];
Accuracy1V1Mat=[];
Accuracy1VATrainMat=[];
Accuracy1V1TrainMat=[];
crange=0.0001:40:2000;
grange=0.001:0.02:0.4;

%Looping through various parameters
for cvalue = crange
    
    %Param initialisation
    Accuracy1VA=[];
    Accuracy1V1=[];
    Accuracy1VATrain=[];
    Accuracy1V1Train=[];
    
    for gvalue=grange
    
    %String to set svm params
    tValueStr='-t 2';
    cValueStr=[' -c ',mat2str(cvalue)];
    gValueStr=[' -g ',mat2str(gvalue)];
    param=strcat(tValueStr,cValueStr,gValueStr);
    
    %Param initialisation
    score=[];
    class=10;
    LabelIdx=size(data_train,2);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %1vA algotithm on testing data
    for i=1:class
        
        %Label one class as +1 and the rest as -1
        LabelMat=2*double(data_train(:,LabelIdx)==i)-1;
        
        %Train svm
        SVMModel = svmtrain(LabelMat,data_train(:,[1:LabelIdx-1]),param);
        
        %Test model
        [predict_label, accuracy, prob_estimates]=svmpredict(data_test(:,LabelIdx),data_test(:,[1:LabelIdx-1]),SVMModel);
        
        %Store prediction probability
        score=[score,prob_estimates];
    end
    
    %Find the max probability and save it
    [~,result]=max(score,[],2);
    
    %Find the testing accuracy
    Accuracy1VA=[Accuracy1VA;sum(double(result==data_test(:,LabelIdx)))/length(result)];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %1v1 algorithm
    
    %Param initialisation
    voteMat=[];
    
    %Looping through class
    for i=1:class-1
        for j=i+1:class
            
            %Select two classes of data
            data_train_1v1=data_train(data_train(:,LabelIdx)==i | data_train(:,LabelIdx)==j,:);
            %Label one class as 1 and the other class as -1
            LabelMat1v1=2*double(data_train_1v1(:,LabelIdx)==i)-1;
            
            %Model training
            SVMModel1v1 = svmtrain(LabelMat1v1,data_train_1v1(:,[1:LabelIdx-1]),param);
            
            %Model testing
            [predict_label_1v1, accuracy1v1, prob_estimates_1v1]=svmpredict(data_test(:,LabelIdx),data_test(:,[1:LabelIdx-1]),SVMModel1v1);
            
            %Find all vote and save to a matrix
            vote=(predict_label_1v1+1)/2*i+(-predict_label_1v1+1)/2*j;
            voteMat=[voteMat,vote];
        end
    end
    
    %Find the class with the highest vote
    result1v1=mode(voteMat,2);
    
    %Calculate accuracy
    Accuracy1V1=[Accuracy1V1;sum(double(result1v1==data_test(:,LabelIdx)))/length(result)];

    %find confusion matrix
    cfm1vA=confusionmat(result,data_test(:,LabelIdx));
    cfm1v1=confusionmat(result1v1,data_test(:,LabelIdx));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Same process has been repeated on the training data 
    
    %Trainning accuracy
    score=[];
    class=10;
    for i=1:class
        LabelMat=2*double(data_train(:,LabelIdx)==i)-1;
        SVMModel = svmtrain(LabelMat,data_train(:,[1:LabelIdx-1]),param);
        [predict_label, accuracy, prob_estimates]=svmpredict(data_train(:,LabelIdx),data_train(:,[1:LabelIdx-1]),SVMModel);
        score=[score,prob_estimates];
    end
    [~,result]=max(score,[],2);
    Accuracy1VATrain=[Accuracy1VATrain;sum(double(result==data_test(:,LabelIdx)))/length(result)];

    voteMat=[];
    for i=1:class-1
        for j=i+1:class
            data_train_1v1=data_train(data_train(:,LabelIdx)==i | data_train(:,LabelIdx)==j,:);
            LabelMat1v1=2*double(data_train_1v1(:,LabelIdx)==i)-1;
            SVMModel1v1 = svmtrain(LabelMat1v1,data_train_1v1(:,[1:LabelIdx-1]),param);
            [predict_label_1v1, accuracy1v1, prob_estimates_1v1]=svmpredict(data_train(:,LabelIdx),data_train(:,[1:LabelIdx-1]),SVMModel1v1);
            vote=(predict_label_1v1+1)/2*i+(-predict_label_1v1+1)/2*j;
            voteMat=[voteMat,vote];
        end
    end
    
    result1v1=mode(voteMat,2);
    Accuracy1V1Train=[Accuracy1VATrain;sum(double(result1v1==data_test(:,LabelIdx)))/length(result)];
    end
    Accuracy1VAMat=[Accuracy1VAMat,Accuracy1VA];
    Accuracy1V1Mat=[Accuracy1V1Mat,Accuracy1V1];
    Accuracy1VATrainMat=[Accuracy1VATrainMat,Accuracy1VATrain];
    Accuracy1V1TrainMat=[Accuracy1V1TrainMat,Accuracy1V1Train];
    
end

figure;
subplot(1,2,1);
Xtemp=repmat(grange',[1 size(Accuracy1V1Mat,2)]);
Ytemp=repmat(crange,[size(Accuracy1V1Mat,1) 1]);
%X=reshape(Xtemp,size(Accuracy1VAMat,1)*size(Accuracy1VAMat,2),1);
%Y=reshape(Ytemp,size(Accuracy1VAMat,1)*size(Accuracy1VAMat,2),1);
Z=reshape(Accuracy1V1Mat,size(Accuracy1V1Mat,1)*size(Accuracy1V1Mat,2),1);
%scatter3(X,Y,Z);
%figure;
xs=crange;
ys=grange;
zs=Accuracy1V1Mat;
surf(xs,ys,zs);
xlabel('c value');
ylabel('g value');
zlabel('Accuracy');
title('One against one accuracy graph');
[value1,index1]=max(Accuracy1V1Mat(:));
[row1,col1] = find(Accuracy1V1Mat==value1)

%figure;
%1vA
Xtemp=repmat(grange',[1 size(Accuracy1VAMat,2)]);
Ytemp=repmat(crange,[size(Accuracy1VAMat,1) 1]);
%X=reshape(Xtemp,size(Accuracy1V1Mat,1)*size(Accuracy1V1Mat,2),1);
%Y=reshape(Ytemp,size(Accuracy1V1Mat,1)*size(Accuracy1V1Mat,2),1);
Z=reshape(Accuracy1VAMat,size(Accuracy1VAMat,1)*size(Accuracy1VAMat,2),1);
%scatter3(X,Y,Z);
subplot(1,2,2);
xs=crange;
ys=grange;
zs=Accuracy1VAMat;
surf(xs,ys,zs);
xlabel('c value');
ylabel('g value');
zlabel('Accuracy');
title('One against all accuracy graph');
[value2,index2]=max(Accuracy1VAMat(:));
[row2,col2] = find(Accuracy1VAMat==value2)

%

figure;
subplot(1,2,1);
hmo=HeatMap(cfm1vA,'Colormap','redbluecmap','RowLabels',[1:10],'ColumnLabels',[1:10]);
addTitle(hmo,'Confusion Matrix for the one against one algorithm');

subplot(1,2,2);
hmo=HeatMap(cfm1v1,'Colormap','redbluecmap','RowLabels',[1:10],'ColumnLabels',[1:10]);
addTitle(hmo,'Confusion Matrix for the one against all algorithm, polynomial kernel');


figure;
subplot(1,2,1)
plot(crange,Accuracy1V1Mat);
title('One against one accuracy graph');
xlabel('C value');
ylabel('Accuracy');

subplot(1,2,2)
plot(crange,Accuracy1VAMat);
title('One against all accuracy graph');
xlabel('C value');
ylabel('Accuracy');

meanArray=[];
for class=1:10
    idx=15*(class-1)+1:15*class;
    [B,I]=sort(score(idx,:),2,'descend');
    [r,c]=find(I'==class);
    meanArray=[meanArray,r];
end
mr1VA=mean(mean(meanArray));

%mean rank for votemat
mArray=[];
for i=1:150
    [mtemp,center]=hist(voteMat(i,:),1:15);
    mArray=[mArray;mtemp];
end
meanArray=[];
for class=1:10
    idx=15*(class-1)+1:15*class;
    [B,I]=sort(mArray(idx,:),2,'descend');
    [r,c]=find(I'==class);
    meanArray=[meanArray,r];
end
mr1V1=mean(mean(meanArray));