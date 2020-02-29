clc
clear all;


% training and testing data set
x_train_set = (-1:0.05:1);
y_train = 1.2 * sin(pi*x_train_set) - cos(2.4*pi*x_train_set);

x_test_set = (-1:0.01:1);
y_test = 1.2 * sin(pi*x_test_set) - cos(2.4*pi*x_test_set);

epochs = 100;

for n = [1:10,20,50,100]
    [net, accu_train] = train_seq(n, x_train_set, y_train,length(x_train_set), epochs);

    % get predict results for both train and test dats set
    x_train_pred = sim(net,x_train_set);
    x_test_pred = sim(net,x_test_set);

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

function[ net, accu_train ] = train_seq( n, inputs, labels, train_num, epochs )
%% Construct a 1-n-1 MLP and conduct sequential training.
%
% Args:
% n: int, number of neurons in the hidden layer of MLP.
% inputs: matrix of (input_dim, input_num), containing possibly preprocessed input data as input.
% labels: vector of (1, input_num), containing corresponding label of each input.
% train_num: int, number of training inputs.
% val_num: int, number of validation inputs.
% epochs: int, number of training epochs.


% Returns:
% net: object, containing trained network.
% accu_train: vector of (epochs, 1), containing the accuracy on training set of each eopch during training
% accu_val: vector of (epochs, 1), containing the accuracy on validation set of each eopch during training.

%% 1. Change the input to cell array form for sequential training
inputs_c = num2cell(inputs, 1);
labels_c = num2cell(labels, 1);

%% 2. Construct and configure the MLP
net = fitnet(n);
net.divideFcn = 'dividetrain'; % input for training only
net.performParam.regularization = 0.25; % regularization strength
net.trainFcn = 'traingdx'; % 'trainrp' 'traingdx'
net.trainParam.epochs = epochs;
accu_train = zeros(epochs,1); % record accuracy on training set of each epoch


%% 3. Train the network in sequential mode
for i = 1 : epochs
    %display(['Epoch: ', num2str(i)])
    idx = randperm(train_num); % shuffle the input
    net = adapt(net, inputs_c(:,idx), labels_c(:,idx));
    pred_train = round(net(inputs(:,1:train_num))); % predictions on training set
    accu_train(i) = 1 - mean(abs(pred_train-labels(1:train_num)));
    
end

end