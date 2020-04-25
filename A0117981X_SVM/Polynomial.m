close all;  
clear all;
clc;

% Load Data  
% train_data: [57×2000 double]  
% train_label: [2000×1 double]
load('train.mat');
load('test.mat');
[Feature,DataSize] = size(train_data);
% Data Standardization
x_mean = mean(train_data,2);
x_std = std(train_data,0,2);
x_train = (train_data - x_mean)./x_std;
x_test = (test_data -x_mean)./x_std;

%C =1e6; % hard margin
C = 2.1; %soft margin, change value accoringly to see results
P_set = [1,2,3,4,5];
for p= P_set
    
    % check the Mercer's Condition using the Gram Matrixs 
    K = (x_train'*x_train+1).^p;
    e = eig(K);
    min_e = min(e);
    if min_e < -3e-2  %-0.1
        disp('Kernel failed Mercer’s  condition');
    else
        disp('Kernel satisfies Mercer’s condition')
        % Meet Mercer condition
        %Quadratic Programming
        H = zeros(DataSize,DataSize);
        for i=1:DataSize
            for j=1:DataSize
                G_matrix = (x_train(:,i)'*x_train(:,j)+1)^p;
                H(i,j) = train_label(i)*train_label(j)*G_matrix;
            end
        end
        f = -1*ones(DataSize,1);
        A = [];
        b = [];
        Aeq=train_label';
        beq = 0;
        lb = zeros(DataSize,1);
        ub = C * ones(DataSize,1) ;
        x0 = [];
        threshold=1e-4;
        options = optimset('LargeScale','off','MaxIter',1000);
        alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
        idx=find(alpha>threshold);
        b =0;
         for i=idx(:)'
                z=zeros(DataSize,1);
                for j=1:DataSize
                    z(j)=alpha(j).*train_label(j).*(x_train(:,j)'*x_train(:,i)+1).^p;
                end
                b = b + train_label(i)-sum(z);
         end
         bo = b/length(idx);
         
       %train set
       g_train=zeros(DataSize,1);
       for i=1:DataSize
          z =zeros(DataSize,1);
          for j=1:DataSize
              z(j)=alpha(j).*train_label(j).*(x_train(:,j)'*x_train(:,i)+1).^p;
          end
          g_train(i)=sum(z)+bo;
       end
       train_predicted=sign(g_train);
       total = length(train_label);
       correct = length(find((train_label-train_predicted)==0));        
       accuracy_train= correct/total;      
       disp(['p=',num2str(p),',c=',num2str(C),',accuracy_train=',num2str(accuracy_train)]);
       
       %test
        g_test=zeros(length(test_label),1);
        for i=1:length(test_label)
            z=zeros(DataSize,1);
            for j=1:DataSize
                z(j)=alpha(j).*train_label(j).*(x_train(:,j)'*x_test(:,i)+1).^p;
            end
            g_test(i)=sum(z)+bo;
        end
        test_predicted=sign(g_test);
        total = length(test_label);
        correct = length(find((test_label-test_predicted)==0));        
        accuracy_test= correct/total;
        disp(['p=',num2str(p),',c=',num2str(C),',accuracy_test=',num2str(accuracy_test)]);
        % ROC curve on test set
        result1 = plot_roc( test_predicted, test_label,C,p );
        %disp(result1);
    end   
end

