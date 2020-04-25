close all;  
clear all;
clc;

% Load Data  
% train_data: [57×2000 double]  
% train_label: [2000×1 double]
load('train1.mat');
load('test1.mat');
[Feature,DataSize] = size(train_data);

% Data Standardization
x_mean = mean(train_data,2);
x_std = std(train_data,0,2);
x_train = (train_data - x_mean)./x_std;
x_test = (test_data -x_mean)./x_std;

% check the Mercer's Condition using the Gram Matrixs 
K = x_train'*x_train;
eigenValue = eig(K);
min_e = min(eigenValue);

if min_e < -0.0001
    disp('Kernel failed Mercer condition');
else
    disp('Kernel satisfies Mercer’s condition')
    % Meet Mercer condition
    %Quadratic Programming
    H = zeros(DataSize,DataSize);
    for i=1:DataSize
        for j=1:DataSize
            G_matrix = x_train(:,i)'*x_train(:,j);
            H(i,j) = train_label(i)*train_label(j)*G_matrix;
        end
    end
  
    f = -1*ones(DataSize,1);
    C = 1000;
    A = [];
    b = [];
    Aeq=train_label';
    beq = 0;
    lb = zeros(DataSize,1);
    ub = C*ones(DataSize,1);
    x0 = [];
    threshold=1e-4;
    options = optimset('LargeScale','off','MaxIter',1000);
    alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
    idx=find(alpha>threshold);
    wo = 0;
    b = 0; 
    for i=1:DataSize
        wo = wo + alpha(i,1).*train_label(i,1).*x_train(:,i);
    end
    
    for i= idx(:)' 
        b = b + (1/train_label(i))-wo'*x_train(:,i);
    end
    bo = b/length(idx);
    
    %Accuracy on train set
    g_train=wo'*x_train+bo;
    train_predicted=sign(g_train);
    total = length(train_label);
    correct = length(find((train_label-train_predicted')==0));        
    accuracy_train= correct/total;   
    disp(['c=',num2str(C),',accuracy_train=',num2str(accuracy_train)]);
    
    %Accuracy on test set
    g_test=wo'*x_test+bo;
    test_predicted=sign(g_test);
    total = length(test_label);
    correct = length(find((test_label-test_predicted')==0));        
    accuracy_test= correct/total;
    disp(['c=',num2str(C),',accuracy_test=',num2str(accuracy_test)]);
    
end  

% ROC curve on test set
result1 = plot_roc( test_predicted, test_label,C ,0);
%disp(result1);








