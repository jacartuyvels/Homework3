%% Homework 3 Optimization run.m

data = load('Homework-3-data.mat');

training = data.training;
test = data.test;



label = training.labels;
image = training.images;
shape = size(image);

image = reshape(image(:,:,:),[shape(1)*shape(2), shape(3)]);
[data0 data1] = partition(image,label,0);


%% 1.2 Gradient descent with logistic loss function 

% variable [h_i c] for i=1, ... , 28^2
x= ones(shape(1)*shape(2)+1,1);

tol = 0;
nIter = 60;
R = 2;

x = gradient1(x,tol,nIter,R,data0,data1);

nA = size(data0);
nB = size(data1);
y = zeros(nA(2),1);
y2 = zeros(nB(2),1);

for i = 1:nA(2)
    y(i) = x(1:end-1)'*data0(:,i) + x(end);
end

for i = 1:nB(2)
    y2(i) = x(1:end-1)'*data1(:,i) + x(end) ; 
end

    
    
    

