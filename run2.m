%% Homework 3 optimization
%% Exercise 2

data = load('Homework-3-data.mat');

training = data.training;
test = data.test;



label = training.labels;
image = training.images;
shape = size(image);


image = reshape(image(:,:,:),[shape(1)*shape(2), shape(3)]);
[data0 data1] = partition(image,label,0);

label2 = test.labels;
image2 = test.images;
shape2 = size(image2);


image2 = reshape(image2(:,:,:),[shape2(1)*shape2(2), shape2(3)]);


data0 = data0(:,1:2000);
data1 = data1(:,1:12000);

[h,c]=marginLin(data0,data1,10);

[data0 data1] = partition(image2,label2);

y1 = data0'*h +c
y2 = data1'*h + c
accuracy1 = 0
accuracy2 = 0
for i = 1:size(data0,2)
    accuracy1 = accuracy1 + max(0,y1(i))/y1(i);
end
accuracy1 = accuracy1/(size(data0,2))
 for i = 1:size(data1,2)
    accuracy2 = accuracy2 + min(0,y2(i))/y2(i);
end
accuracy2 = accuracy2/(size(data1,2))


pause;


    
