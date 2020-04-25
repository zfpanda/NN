function  auc = plot_roc( predict, ground_truth,C,p  )  
% INPUTS  
%  predict       
%  ground_truth    
% OUTPUTS  
%  auc            - Return area under ROC curve  
  
%initial position（1.0, 1.0）  
x = 1.0;  
y = 1.0;  
% claculate positive and negative samples 
pos_num = sum(ground_truth==1);  
neg_num = sum(ground_truth==-1);  
 
x_step = 1.0/neg_num;  
y_step = 1.0/pos_num;  

[predict,index] = sort(predict);  
ground_truth = ground_truth(index);  

for i=1:length(ground_truth)  
    if ground_truth(i) == 1  
        y = y - y_step;  
    else  
        x = x - x_step;  
    end  
    X(i)=x;  
    Y(i)=y;  
end  
fig = figure(p+1);      
plot(X,Y,'-bo','LineWidth',0.2,'MarkerSize',1);  
xlabel('False Positive Rate');  
ylabel('True Positive Rate');
if p == 0
title(['ROC Curve C = ',num2str(C)]);  
saveas(fig, ['ROC Curve C = ',num2str(C),'.png'])
else 
title(['ROC Curve C = ',num2str(C),' p = ',num2str(p)]); 
end
auc = -trapz(X,Y);
legend(['AUC =',num2str(auc)],'Location','southeast');
saveas(fig, ['ROC Curve C = ',num2str(C),' p = ',num2str(p),'.png']) 
end  