close all;  
clear all;
clc;

% Load Data  
% train_data: [57×2000 double]  
% train_label: [2000×1 double]
load('train.mat');
%load('test.mat');
load('eval.mat')
[Feature,DataSize] = size(train_data);
% Data Standardization
x_mean = mean(train_data,2);
x_std = std(train_data,0,2);
x_train = (train_data - x_mean)./x_std;
%x_test = (test_data -x_mean)./x_std;
x_eval = (eval_data - x_mean)./x_std;


gamma = 0.002;
%C_set = [0.5,1,10,20,50,100,200,500];
C_set = [500];
for C= C_set
    K = zeros(DataSize,DataSize); %Gram Matrix
    for i = 1:DataSize
        for j = 1 :DataSize
            K(i,j) = exp(-gamma*norm(x_train(:,i) - x_train(:,j))^2);
        end
    end
    % check the Mercer's Condition using the Gram Matrixs 
    e = eig(K);
    min_e = min(e);
    if min_e < -0.0001 %-0.1
        disp('Kernel failed Mercer’s  condition');
    else
        disp('Kernel satisfies Mercer’s condition')
        % Meet Mercer condition
        %Quadratic Programming
        H = zeros(DataSize,DataSize);
        for i=1:DataSize
            for j=1:DataSize
                G_matrix =  exp(-gamma*norm(x_train(:,i) - x_train(:,j))^2);
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
        b = 0;
        for i=idx(:)'
                z=zeros(DataSize,1);
                for j=1:DataSize
                    z(j)=alpha(j).*train_label(j).*exp(-gamma*norm(x_train(:,j) - x_train(:,i))^2);
                end
                b = b + train_label(i)-sum(z);
        end
         bo = b/length(idx);
         
       %train set
       g_train=zeros(DataSize,1);
       for i=1:DataSize
          z =zeros(DataSize,1);
          for j=1:DataSize        
              z(j)=alpha(j).*train_label(j).*exp(-gamma*norm(x_train(:,j) - x_train(:,i))^2);
          end
          g_train(i)=sum(z)+bo;
       end
       train_predicted=sign(g_train);
       total = length(train_label);
       correct = length(find((train_label-train_predicted)==0));        
       accuracy_train= correct/total;   
       disp(['gamma=',num2str(gamma),',c=',num2str(C),',accuracy_train=',num2str(accuracy_train)]);
       
       %test
%         g_test=zeros(length(test_label),1);
%         for i=1:length(test_label)
%             z=zeros(DataSize,1);
%             for j=1:DataSize
%                 %z(j)=alpha(j).*train_label(j).*exp(sum((x_train(:,j) - x_test(:,i)).^2)/(-0.5*sigma^2));
%                 z(j)=alpha(j).*train_label(j).*exp(-gamma*norm(x_train(:,j) - x_test(:,i))^2);
%             end
%             g_test(i)=sum(z)+bo;
%         end
%         test_predicted=sign(g_test);
%         total = length(test_label);
%         correct = length(find((test_label-test_predicted)==0));        
%         accuracy_test= correct/total;
%         disp(['gamma=',num2str(gamma),',c=',num2str(C),',accuracy_test=',num2str(accuracy_test)]);
        
        % eval
        g_test=zeros(length(eval_label),1);
        for i=1:length(eval_label)
            z=zeros(DataSize,1);
            for j=1:DataSize        
                z(j)=alpha(j).*train_label(j).*exp(-gamma*norm(x_train(:,j) - x_eval(:,i))^2);
            end
            g_test(i)=sum(z)+bo;
        end
        eval_predicted=sign(g_test);
        total = length(eval_label);
        correct = length(find((eval_label-eval_predicted)==0));        
        accuracy_eval=correct/total;
        disp(['gamma=',num2str(gamma),',c=',num2str(C),',accuracy_eval=',num2str(accuracy_eval)]);
        
    end   
end




