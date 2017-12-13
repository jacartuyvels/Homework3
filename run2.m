%% Homework 3 optimization
%% Exercise 2

ata = load('Homework-3-data.mat');

training = data.training;
test = data.test;



label = training.labels;
image = training.images;
shape = size(image);

image = reshape(image(:,:,:),[shape(1)*shape(2), shape(3)]);
[data0 data1] = partition(image,label,0);

data0 = data0(:,1:1000);
data1 = data1(:,1:6000);

[h,c,tau]=marginLin(data0, data1);
