close all;  
clear all;
clc;

%load data
%load('task1.mat');
load('qeval.mat');
reward = qevalreward;
%[state,action]=size(reward);
[state,action]=size(reward);

gamma = 0.95;  % discount factor
max_trials = 3000;
Q = zeros(state, action); % Q function initialization 
Q_last = zeros(state, action); % updated Q for every trail 
tic
for i = 1 : max_trials   
    s = 1;
    k = 1;
    Q_last = Q;
    
    while (s ~= 100)
        %greedy policy
        %epsilon = 1/k;
        %epsilon = 100/(100+k); 
        epsilon = 0.6;
        %epsilon = (1+ log(k))/k;
        %epsilon = (1+ 5*log(k))/k;
        % learning rate
        alpha = 1500/(1500+k);  % learning rate
        if alpha < 0.005
                break;
        end
        
        r = rand; %generate random number between(0,1) 
        q_max = max(Q(s,:));
        valid_action = find(reward(s,:)~=-1);
        while 1           
            if r >= epsilon
                act_idx = find(Q(s,:) == max(Q(s,:)));
                idx = randperm(length(act_idx),1);
                current_action = act_idx(idx);
            else
                if q_max == 0
                    idx = randperm(length(valid_action),1);
                    current_action = valid_action(idx); 
                
                else
                    act_idx = find(Q(s,valid_action) ~= max(Q(s,valid_action)));
                    idx = randperm(length(act_idx),1);
                    current_action = valid_action(act_idx(idx));
                end 
            end
            if (reward(s, current_action) >= 0)
                % action available
                break;
            end
        end
        
       
    % choose action
    next_state = move(s, current_action);
    next_reward = reward(s, current_action);
    % update Q function
    Q(s, current_action) = Q(s, current_action) + epsilon * (next_reward + gamma * max(Q(next_state,:)) - Q(s, current_action));    
    s = next_state;
    k = k + 1;    
     
        
    end
    if Q == Q_last
        % converage
        break;
    end
            
end
toc

% greedy search find optimal policy
policy = zeros(2, 19);
s = 1;
t = 1;
total_reward = 0;
while( s ~= 100 )
    act_ls = find(Q(s,:) == max(Q(s,:)));
    idx = randperm(length(act_ls),1);
    a = act_ls(idx);
    policy(1, t) = s;
    policy(2, t) = a;
    total_reward = total_reward + gamma.^(t-1)*reward(s, a);
    %total_reward = total_reward + reward(s, a);
    s = move(s, a);
    policy(1, t+1) = s;
    t = t + 1;
    if t > 19
        break;
    end

end

% output: states and reward
states = policy(1,:);
disp('States:');
disp(states);
disp('Total reward:');
disp(total_reward);

% plot robot trajectory
j = 0:1:10;
i = 0:1:10;
[X,Y] = meshgrid(i,j);
plot(X,Y,'b');
hold on;
plot(X',Y','b');
hold on;
for i = 1: 19
    if policy(1, i) == 0
        break;
    end
    x = floor(policy(1, i)/10) + 0.5;
    y = rem(policy(1, i), 10) - 0.5;
    y = 10 - y;
    if policy(1, i) == 100
        scatter(9.5,0.5,80,'p','k');
    elseif policy(2, i) == 1
        scatter(x,y,60,'^','r');
    elseif policy(2, i) == 2
        scatter(x,y,60,'>','r');
    elseif policy(2, i) == 3
        scatter(x,y,60,'v','r');
    elseif policy(2, i) == 4
        scatter(x,y,60,'<','r');
    end
    hold on;
end
grid on;
hold off;
    

% get next state when moving from current state and take action
function next_state = move(current_state, action)
    switch action
        case 1
            next_state = current_state - 1;
        case 2
            next_state = current_state + 10;
        case 3
            next_state = current_state + 1;
        case 4
            next_state = current_state - 10;
    end
end