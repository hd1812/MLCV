%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MLCV Coursework 1 : Q4 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
close all;

%%%%%%%%%%%% Load Faces %%%%%%%%%%%%
load face.mat

% Eg : each row is one image, to view reshape to 56x46
% imshow(uint8(reshape(X(:,1),56,46))); % First Image as example
% show_face(X(:,1)); % First Image

%%%%%%%%%%%% Partition Data %%%%%%%%%%%%
N = 520;
k = 10;
training_N = (k-1)*N/10;
testing_N = N-training_N;

label = ones(k,52);

% for kfold = 1:k
kfold = 1;

    partition_obj = cvpartition(l,'Kfold',k);
    training_set = X(:,training(partition_obj,kfold)); % 1 indicates the first process, change this number for different training set
    testing_set = X(:,test(partition_obj,kfold));

    %%%%%%%%%%%% PCA on Training Data %%%%%%%%%%%
    M=100;
    mean_face = [];
    eigen_faces = [];
    w = [];
    tic
    for class = 1:52 
        [mean_face(:,class), tmp_eigen_faces, tmp_w] = pca(training_set(:,9*(class-1)+1:9*class),M);
        tmp_eigen_faces = reshape(tmp_eigen_faces,2576*M,1);
        tmp_w = reshape(tmp_w,M*9,1);
        eigen_faces(:,class) = tmp_eigen_faces;
        w(:,class) = tmp_w;
        %[mean_face(class), eigen_faces(class), w(class)] = pcaATA(training_set(:,9*(class-1)+1:9*class),M);
    end
    PCAtime = toc
    
    tic
    %% Step 1 : Normalise test image
    phi = testing_set - repmat(mean_face,1,N/k);

    %% Step 2 & 3 : Project on Eigenspace
    new_w = (phi'*eigen_faces)';

    %% Step 4 : Find error
    tmp_new_w = repelem(new_w,1,training_N);
    tmp_w = repmat(w,1,52);
    error = reshape(sqrt(sum((tmp_new_w - tmp_w).^2)),468,52)';
    
    for x=1:52
        [tmp_sum, min_e] = min(error(x,:));
        label(kfold,x) = ceil(min_e/(k-1));
    end
    RESTtime = toc

% end

%% Step 5 : Accuracy
answer = [1:52];
accuracy = (52-nnz(mode(label)-answer))*100/(52)
% accuracy = (52-nnz((label)-answer))*100/(52)
