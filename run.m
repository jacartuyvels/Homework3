%% Homework 3 Optimization run.m

data = load('Homework-3-data.mat');

training = data.training;
test = data.test;



label = training.labels;
image = training.images;
shape = size(image);

image = reshape(image(:,:,:),[shape(1)*shape(2), shape(3)]);
shape = size(image);
x = ones(shape(1)+2,1);
tol = 0;
nIter = 10;
R = 2;
[x,y] = gradient1(x,tol,nIter,R,image,label);

plot(y,(1:11))