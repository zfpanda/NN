clc
clear all;


% training and testing data set
x_train_set = (-1:0.05:1);
y_train = 1.2 * sin(pi*x_train_set) - cos(2.4*pi*x_train_set);

x_test_set = (-1:0.01:1);
y_test = 1.2 * sin(pi*x_test_set) - cos(2.4*pi*x_test_set);

train_func = 'trainlm';
epochs = 50;

for n = [1:10,20,50,100]
    %[net, accu_train] = train_seq(n, x_train_set, y_train,length(x_train_set), epochs);
    % build model
    net = patternnet(n);
    net.divideFcn = 'dividetrain';
    net.performFcn = 'mse';
    net.trainFcn = train_func;

    % train model
    net = train(net, x_train_set, y_train);
    % get predict results for both train and test dats set
    x_train_pred = net(x_train_set);
    x_test_pred = net(x_test_set);

    fig = figure(n);
    %set(gcf,'PaperPositionMode','auto')
    %set(fig, 'Position', [200 200 500 400])
    set(gcf,'unit','normalized','position',[0.2,0.2,0.64,0.32]);
    % show ploton training set
    subplot(1,2,1);
    p = plot(x_train_set,y_train,'*',x_train_set,x_train_pred,'g-');
    p(1).MarkerSize = 4;
    p(2).LineWidth = 2;
    title(['MLP 1-',num2str(n),'-1; Training set']);
    xlabel('x');
    ylabel('y');
    legend('Output','Predict')
    xlim([-3 3]);

    % show ploton test set
    subplot(1,2,2);
    p = plot(x_test_set,y_test,'*',x_test_set,x_test_pred,'r-');
    p(1).MarkerSize = 4;
    p(2).LineWidth = 2;
    title(['MLP 1-',num2str(n),'-1; Test set']);
    xlabel('x');
    ylabel('y');
    legend('Output','Predict')
    xlim([-3 3]);

    saveas(fig, ['q2a_n',num2str(n),'.png'])
    pred_x3 = net([-3, 3]);
    display(['n = ',num2str(n),' results for x = -3 and x = 3: ', num2str(pred_x3)]);
end

