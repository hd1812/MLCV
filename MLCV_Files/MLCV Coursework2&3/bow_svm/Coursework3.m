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
% Select and load dataset
[data_train, data_test] = getData('Toy_Spiral'); % {'Toy_Gaussian', 'Toy_Spiral', 'Toy_Circle', 'Caltech'}


%%%%%%%%%%%%%
% check the training data
    % data_train(:,1:2) : [num_data x dim] Training 2D vectors
    % data_train(:,3) : [num_data x 1] Label of training data, {1,2,3}
    
plot_toydata(data_train);

%Bootstrap
k=100;
data_train_bagging=[];
for i=1:4
    data=[datasample(data_train,k),ones(k,1)*i];
    data_train_bagging=[data_train_bagging;data];
end;
%}
%{
%Weak trainer setup, id=1
opts= struct;
opts.depth= 2; %tree depth
opts.numTrees= 1;
opts.numSplits= 5;  %Number of splits to try
opts.verbose= true;
opts.classifierID= 3; % weak learners to use. Can be an array for mix of weak learners too

rows=data_train_bagging(:,4)==1;
data_train_temp1to3=data_train_bagging(rows,1:3);
data_train_temp3=data_train_bagging(rows,3);

%find the information gain for weak learners
Iarray=[]
for s=1:120
opts.numSplits= s;
[m0,i]=weakTrain(data_train_bagging(rows,1:2), data_train_bagging(rows,3), opts);
Iarray=[Iarray,i];
end
plot(Iarray);
grid on;
title('Information gain against the degree of randomness');

m1= treeTrain(data_train_bagging(rows,1:2), data_train_bagging(rows,3), opts);

%visualisation of a weak node id=1
figure('Position',[100 100 900 250]);
subplot(2,2,1);
learnerVis(3,m1.weakModels{1, 1},data_train_temp1to3);
title('Visualisation of conic section split function');
subplot(2,2,2);
hist(data_train_temp3,1:3);
title('Histogram of the training set');
subplot(2,2,3);
bar(m1.leafdist(1,:));
title('Histogram of the left node');
subplot(2,2,4);
bar(m1.leafdist(2,:));
title('Histogram of the right node');

%Weak trainer 2 setup id=2
opts= struct;
opts.depth= 2; %tree depth
opts.numTrees= 1;
opts.numSplits= 10;  %Number of splits to try
opts.verbose= true;
opts.classifierID= 2; % weak learners to use. Can be an array for mix of weak learners too

rows=data_train_bagging(:,4)==1;
m1= treeTrain(data_train_bagging(rows,1:2), data_train_bagging(rows,3), opts);

%visualisation of a weak node id=2
figure('Position',[100 100 600 250]);
subplot(1,2,1);
bar(m1.leafdist(1,:));
title('Histogram of the left node');
subplot(1,2,2);
bar(m1.leafdist(2,:));
title('Histogram of the right node');
%}
%%
%{
%grow forest to test accuracy
%tree setup
opts= struct;
opts.depth= 5; %tree depth
opts.numTrees= 1;
opts.numSplits= 10;  %Number of splits to try
opts.verbose= true;
opts.classifierID= 4; % weak learners to use. Can be an array for mix of weak learners too

rows=data_train_bagging(:,4)==1;
data_train_temp1to3=data_train_bagging(rows,1:3);
data_train_temp3=data_train_bagging(rows,3);
data_train_set=data_train_bagging(rows,:);

m1= forestTrain(data_train_bagging(rows,1:2), data_train_bagging(rows,3), opts);

%find training accuracy
figure('Position',[100 -500 800 3000]);
subplot(3,2,1);
hold on
DTTrain = forestTest(m1, data_train_set(:,1:2));
data_train_temp=data_train_set;
data_train_temp(:,3)=DTTrain;
plot_toydata(data_train_temp(:,1:3));
trainingAccuracy=size(DTTrain(DTTrain==data_train_set(:,size(data_train_set,2))),1)/size(DTTrain,1);
%find testing image
subplot(3,2,2,'Color','w');
DTTrain = forestTest(m1, data_test(:,1:2));
data_test_temp=data_test;
data_test_temp(:,3)=DTTrain;
plot_toydata(data_test_temp);
plot_toydata(data_train_temp1to3);

%visualisation of a weak node
for i=1:4
    subplot(3,2,i+2);
    bar(m1.treeModels{1, 1}.leafdist(i,:));
    grid on;
end
%}
%%
%{
%Train a forest to test accuracy
opts= struct;
opts.depth= 5; %tree depth
opts.numTrees= 1;
opts.numSplits= 5;  %Number of splits to try
opts.verbose= true;
opts.classifierID= 1; % weak learners to use. Can be an array for mix of weak learners too

rows=data_train_bagging(:,4)==1;
data_train_temp1to3=data_train_bagging(rows,1:3);
data_train_temp3=data_train_bagging(rows,3);
m1= forestTrain(data_train_bagging(rows,1:2), data_train_bagging(rows,3), opts);
WTTrain = forestTest(m1, data_train_bagging(rows,1:2));
%Decision tree training on various data sets
%}

%{
%Weak trainer setup
opts= struct;
opts.depth= 5; %tree depth
opts.numTrees= 10;
opts.numSplits= 5;  %Number of splits to try
opts.verbose= true;
opts.classifierID= 1; % weak learners to use. Can be an array for mix of weak learners too

%Visualisaion on training set
figure;
for i=1:1
    rows=data_train_bagging(:,4)==i;
    m2= forestTrain(data_train_bagging(rows,1:2), data_train_bagging(rows,3), opts);
    DTTrain = forestTest(m2, data_train(:,1:2));
    data_train_temp=data_train;
    data_train_temp(:,3)=DTTrain;

    %subplot(4,2,2*i-1);
    %figure;
    plot_toydata(data_train_temp);
end

%Visualisaion on testing set
for i=1:4
    rows=data_train_bagging(:,4)==i;
    m2= forestTrain(data_train_bagging(rows,1:2), data_train_bagging(rows,3), opts);
    DTTrain = treeTest(m2, data_test(:,1:2));
    data_test_temp=data_test;
    data_test_temp(:,3)=DTTrain;

    %subplot(4,2,2*i);
    %figure;
    %plot_toydata(data_test_temp);
    %plot_toydata(data_train); 
end
%}
%%
%{
opts= struct;
opts.depth= 9; %tree depth
opts.numTrees= 100;
opts.numSplits= 5;  %Number of splits to try
opts.verbose= true;
opts.classifierID= 1; % weak learners to use. Can be an array for mix of weak learners too

%Train a random forest
m3=forestTrain(data_train(:,1:2),data_train(:,3),opts);

test_point = [-.5 -.7; .4 .3; -.7 .4; .5 -.5];

RFTrain=forestTest(m3, test_point(:,1:2));
data_test_temp=[test_point,zeros(size(test_point,1),1)];
data_test_temp(:,3)=RFTrain;
figure;
plot_toydataS(data_test_temp);
plot_toydata(data_train); 
title('Novel testing point classification results');
%}
%% testing parameters on spiral data
%{
opts= struct;
opts.depth= 5; %tree depth
opts.numTrees= 250;
opts.numSplits= 10;  %Number of splits to try
opts.verbose= true;
opts.classifierID= 5; % weak learners to use. Can be an array for mix of weak learners too
TrainingAccuracyArray=[];
idx=1;

for dt=5:2:9
    for nt=50:100:300
        opts.depth= dt;
        opts.numTrees=nt;
        
        m3=forestTrain(data_train(:,1:2),data_train(:,3),opts);
        
        %test model on training set
        TrainResult = forestTest(m3, data_train(:,1:2));
        TrainAccuracy=size(TrainResult(TrainResult==data_train(:,3)),1)/size(data_train,1);
        TrainingAccuracyArray=[TrainingAccuracyArray,TrainAccuracy];
        
        %test model on testing set
        xrange = [-1.5 1.5];
        yrange = [-1.5 1.5];
        inc = 0.02;
        [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
        image_size = size(x);
        xy = [x(:) y(:)];

        [yhat, ysoft] = forestTest(m3, xy);
        decmap= reshape(ysoft, [image_size 3]);
        decmaphard= reshape(yhat, image_size);

        subplot(122);
        %subplot(3,3,idx);
        imagesc(xrange,yrange,decmaphard);
        hold on;
        set(gca,'ydir','normal');
        cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1];
        colormap(cmap);
        plot(data(data(:,end)==1,1), data(data(:,end)==1,2), 'o', 'MarkerFaceColor', [.9 .5 .5], 'MarkerEdgeColor','k');
        plot(data(data(:,end)==2,1), data(data(:,end)==2,2), 'o', 'MarkerFaceColor', [.5 .9 .5], 'MarkerEdgeColor','k');
        plot(data(data(:,end)==3,1), data(data(:,end)==3,2), 'o', 'MarkerFaceColor', [.5 .5 .9], 'MarkerEdgeColor','k');
        hold off;
        title('Sample result for two-pixel test')
        %title(sprintf('Depth: %d, No. of trees: %d\n', dt, nt));
        
        idx=idx+1;
    end
end
%}




%%
%%Q3
%[data_train, data_test] = getData('Caltech');

%save ws
%load ws
load data_test.mat

%Parameter set up
opts= struct;
opts.depth= 5; %tree depth
opts.numTrees= 250;
opts.numSplits= 10;  %Number of splits to try
opts.verbose= true;
opts.classifierID= 5; % weak learners to use. Can be an array for mix of weak learners too
TrainingAccuracyArray=[];

Result=[];
dtrange=5:11;
ntrange=50:100:350;
for dt=dtrange
    for nt=ntrange
        for ns=10:10
            
            opts= struct;
            opts.depth= dt; %tree depth
            opts.numTrees= nt;
            opts.numSplits= ns;  %Number of splits to try
            opts.verbose= true;

            m3=forestTrain(data_train(:,1:size(data_train,2)-1),data_train(:,size(data_train,2)),opts);
            RFTrain=forestTest(m3, data_train(:,1:size(data_train,2)-1));
            AccuracyTrain=size(RFTrain(RFTrain==data_train(:,size(data_train,2))),1)/size(RFTrain,1);

            RFTest=forestTest(m3, data_test(:,1:size(data_test,2)-1));
            AccuracyTest=size(RFTest(RFTest==data_test(:,size(data_test,2))),1)/size(RFTest,1);
            Result=[Result;dt,nt,ns,AccuracyTrain,AccuracyTest];
            
        end
    end
end

figure;
ResultMat=reshape(Result(:,5),size(ntrange,2),size(dtrange,2));
mesh(dtrange,ntrange,ResultMat);
xlabel('Depth of trees');
ylabel('Number of trees');
zlabel('Prediction accuracy');
title('The prediction accuracy graph of the two-pixel test weak learner');
maxResult=max(Result(:,5));


folderName = './Caltech_101/101_ObjectCategories';
classList = dir(folderName);
classList = {classList(3:end).name} % 10 classes
load imgIdx.mat;
idx=imgIdx(1);
subFolderName = fullfile(folderName,classList{3});
imgList = dir(fullfile(subFolderName,'*.jpg'));
imgIdx_te = imgIdx{1}(16:30);

cfmAM=confusionmat(RFTest,data_test(:,size(data_test,2)));
hmo=HeatMap(cfmAM,'Colormap','redbluecmap','RowLabels',[1:10],'ColumnLabels',[1:10]);
addTitle(hmo,'Confusion Matrix for RF with axis-aligned weak learner');
plot(hmo)

figure;
subplot(121)
I = imread(fullfile(subFolderName,imgList(imgIdx_te(1)).name));
imshow(I);

subplot(122)
I = imread(fullfile(subFolderName,imgList(imgIdx_te(2)).name));
imshow(I);

figure;
subplot(121)
I = imread(fullfile(subFolderName,imgList(imgIdx_te(3)).name));
imshow(I);

subplot(122)
I = imread(fullfile(subFolderName,imgList(imgIdx_te(4)).name));
imshow(I);

figure;
subplot(121)
I = imread(fullfile(subFolderName,imgList(imgIdx_te(5)).name));
imshow(I);

subplot(122)
I = imread(fullfile(subFolderName,imgList(imgIdx_te(6)).name));
imshow(I);
