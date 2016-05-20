clc
clear all
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Pixel clustering through k-means clustering

%Image directory
dir='Caltech_101/101_ObjectCategories/wild_cat/image_0001.jpg';

%Image information
%info=imfinfo(dir)

%Read image
im_temp=double(imread(dir));

%Image Visualisation
figure;
subplot(2,2,1);
imshow(uint8((im_temp)))
title('Original');

%Reshape for k-means function
im = double(reshape(im_temp,size(im_temp,1)*size(im_temp,2),size(im_temp,3)));

%Apply k-means on the image
class=3;
[idx,C,sumd,D]=kmeans(im,class,'Display','iter');

%Find the mean colour for the image
ColourVec=[];
for i =1:size(im,1)
        ColourVec = [ColourVec;C(idx(i),:)];
end
ColourVecNorm=im2double(ColourVec)/256;

subplot(2,2,2);
%Reconstruct images after k-means
imshow(uint8((reshape(ColourVec,size(im_temp)))));
title('K = 3');

%{
%K = 10
%Apply k-means on the image
class=10;
[idx,C,sumd,D]=kmeans(im,class,'Display','iter');

%Find the mean colour for the image
ColourVec=[];
for i =1:size(im,1)
        ColourVec = [ColourVec;C(idx(i),:)];
end
ColourVecNorm=im2double(ColourVec)/256;

%k=20
subplot(2,2,3);
%Reconstruct images after k-means
imshow(uint8((reshape(ColourVec,size(im_temp)))));
title('K = 10');

%Apply k-means on the image
class=20;
[idx,C,sumd,D]=kmeans(im,class,'Display','iter');

%Find the mean colour for the image
ColourVec=[];
for i =1:size(im,1)
        ColourVec = [ColourVec;C(idx(i),:)];
end
ColourVecNorm=im2double(ColourVec)/256;

%Reconstruct images after k-means
subplot(2,2,4);
imshow(uint8((reshape(ColourVec,size(im_temp)))));
title('K = 20');
%}

%Initialisation 1
%Show the movement of centroids
CentrVec=[];
maxiter=20;
sumt=[];
[idx,C,sumd,D]=kmeans(im,class,'Display','final','MaxIter',1);
for iter=1:maxiter-1
    [idx,C,sumd,D]=kmeans(im,class,'Display','final','MaxIter',1,'start',C);
    CentrVec=[CentrVec;C];
    sumtemp=sum(sumd);
    sumt=[sumt;sumtemp];
end

%Find the mean colour for the image
ColourVec=[];
for i =1:size(im,1)
        ColourVec = [ColourVec;C(idx(i),:)];
end
ColourVecNorm=im2double(ColourVec)/256;

figure('Position', [100, 100, 800, 300]);
subplot(1,2,1);
scatter3(im(:,1),im(:,2),im(:,3),0.1,ColourVecNorm);
title('Initialisation 1');

hold on;
for i=1:class
        plot3(CentrVec(i:class:end,1),CentrVec(i:class:end,2),CentrVec(i:class:end,3),'LineStyle','-','LineWidth',3,'Marker','o');
end

subplot(1,2,2);
plot(sumt);
title('Sum of distance against iterations for Initialisation 1');
grid on;

%Initialisation 2
%Show the movement of centroids
CentrVec=[];
maxiter=20;
sumt=[];
[idx,C,sumd,D]=kmeans(im,class,'Display','final','MaxIter',1);
for iter=1:maxiter-1
    [idx,C,sumd,D]=kmeans(im,class,'Display','final','MaxIter',1,'start',C);
    CentrVec=[CentrVec;C];
    sumtemp=sum(sumd);
    sumt=[sumt;sumtemp];
end

%Find the mean colour for the image
ColourVec=[];
for i =1:size(im,1)
        ColourVec = [ColourVec;C(idx(i),:)];
end
ColourVecNorm=im2double(ColourVec)/256;

figure('Position', [100, 100, 800, 300]);
subplot(1,2,1);
scatter3(im(:,1),im(:,2),im(:,3),0.1,ColourVecNorm);
title('Initialisation 2');

hold on;
for i=1:class
        plot3(CentrVec(i:class:end,1),CentrVec(i:class:end,2),CentrVec(i:class:end,3),'LineStyle','-','LineWidth',3,'Marker','o');
end

subplot(1,2,2);
plot(sumt);
title('Sum of distance against iterations for Initialisation 2');
grid on;

%Initialisation 3
%Show the movement of centroids
CentrVec=[];
maxiter=20;
sumt=[];
[idx,C,sumd,D]=kmeans(im,class,'Display','final','MaxIter',1);
for iter=1:maxiter-1
    [idx,C,sumd,D]=kmeans(im,class,'Display','final','MaxIter',1,'start',C);
    CentrVec=[CentrVec;C];
    sumtemp=sum(sumd);
    sumt=[sumt;sumtemp];
end

%Find the mean colour for the image
ColourVec=[];
for i =1:size(im,1)
        ColourVec = [ColourVec;C(idx(i),:)];
end
ColourVecNorm=im2double(ColourVec)/256;

figure('Position', [100, 100, 800, 300]);
subplot(1,2,1);
scatter3(im(:,1),im(:,2),im(:,3),0.1,ColourVecNorm);
title('Initialisation 3');

hold on;
for i=1:class
        plot3(CentrVec(i:class:end,1),CentrVec(i:class:end,2),CentrVec(i:class:end,3),'LineStyle','-','LineWidth',3,'Marker','o');
end

subplot(1,2,2);
plot(sumt);
title('Sum of distance against iterations for Initialisation 3');
grid on;

%%
%Apply k-means on the image
class=3;
randSeed=ceil(rand(3,1)*size(im,1));
randStart=im(randSeed,:);

CentrVec=[];
maxiter=20;
sumt=[];
[idx,C,sumd,D]=kmeans(im,class,'Display','final','MaxIter',1,'Start',randStart,'OnlinePhase','on');
CentrVec=[CentrVec;C];
sumtemp=sum(sumd);
sumt=[sumt;sumtemp];
for iter=1:maxiter-1
    [idx,C,sumd,D]=kmeans(im,class,'Display','final','MaxIter',1,'start',C,'OnlinePhase','on');
    CentrVec=[CentrVec;C];
    sumtemp=sum(sumd);
    sumt=[sumt;sumtemp];
end

%Find the mean colour for the image
ColourVec=[];
for i =1:size(im,1)
        ColourVec = [ColourVec;C(idx(i),:)];
end
ColourVecNorm=im2double(ColourVec)/256;

figure('Position', [100, 100, 800, 300]);
subplot(1,2,1);
scatter3(im(:,1),im(:,2),im(:,3),0.1,ColourVecNorm);
title('Random Initialisation 3');

hold on;
for i=1:class
        plot3(CentrVec(i:class:end,1),CentrVec(i:class:end,2),CentrVec(i:class:end,3),'LineStyle','-','LineWidth',3,'Marker','o');
end

subplot(1,2,2);
plot(sumt);
title('Sum of distance against iterations for Initialisation 3');
grid on;



