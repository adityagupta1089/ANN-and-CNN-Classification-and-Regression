%% clear
clc
%% read input
A = readtable('steering/data.txt', 'ReadVariableNames', 0, 'Delimiter', ...
    '\t');
N = size(A, 1);
X = zeros(1024,N);
Y = zeros(1,N);
wb = waitbar(0,'Please Wait');
for i = 2:N
    waitbar(i/N,wb);
    %% read image and degree
    name = A{i,1}{1,1}(3:end);
    I = imread(['steering/' name]);
    img = double(rgb2gray(I))/255;
    X(:,i) = img(:);
    Y(i) = A{i,2};
end
close(wb);
%% constants
architecture = [1024, 512, 64, 1];
K = length(architecture);
split_ratio = 0.8;
epochs = 5000;
eta = 0.01;
minibatch_size = 64;
%% split data
split = floor(split_ratio * N);
X_train = X(:,1:split);
Y_train = Y(:,1:split);
X_test = X(:,split+1:end);
Y_test = Y(:,split+1:end);
%% weights
ws = cell(1,K-1);
for i = 1:K-1
    ws{i} = 0.02 * rand(architecture(i+1),architecture(i)+1) - 0.01;
end
%% training
N = size(X_train, 2);
iporder = randperm(N);
X = X(:,iporder);
Y = Y(iporder);
for e = 1:epochs
    fprintf('Epoch #%d: ', e);
    for i = 1:minibatch_size:N-minibatch_size
        delta_ws = cell(1,K-1);
        for j = 1:K-1
            delta_ws{j} = zeros(size(ws{j}));
        end
        rng = i:i+minibatch_size;
        x = X_train(:,rng);
        y = Y_train(rng);
        %% forward pass
        vs = cell(1,K);
        vs{1} = [ones(1,size(x,2));x];
        for l = 2:K
            vs{l} = sigmoid(ws{l-1} * vs{l-1});
            if l ~= K
                vs{l} = [ones(1,size(vs{l},2));vs{l}];
            end
        end
        %% gradient calculation
        delta = (vs{K} - y) .* vs{K} .* (1.0 - vs{K});
        delta_ws{K-1} = delta_ws{K-1} + eta * delta * vs{K}';
        %% backward propagation
        for l = K-2:-1:1
            if l == K-2
                delta = ws{l+1}' * delta;
            else
                delta = ws{l+1}' * delta(2:end,:);
            end
            delta_ws{l} = delta_ws{l} + eta * delta(2:end,:) * vs{l}';
        end            
        %% weight updation
        for l = 1:K-1
            ws{l} = ws{l} - delta_ws{l};
        end
    end
    tot_err = 0;
    v = [ones(1,size(X_train,2));X_train];
    for j = 1:K-1
        v = sigmoid(ws{j} * v);
        if j ~= K - 1
           v = [ones(1,size(v,2));v];
        end
    end
    err = sum(sum((Y_train-v).^2))/2;
    fprintf('Error is %f\n', err);
end
%% testing
v = [ones(1,size(X_test,2));X_test];
for j = 1:K-1
    v = sigmoid(ws{j} * v);
    if j ~= K - 1
       v = [ones(1,size(v,2));v];
    end
end
err = sum(sum((Y_test-v).^2))/2;
%% results
fprintf('Total Error is %f\n', err);
%% sigmoid
function val = sigmoid(z)
    val = 1 ./ (1 + exp(-z));
end