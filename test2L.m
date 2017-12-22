clear all
hold off
data = load('Homework-3-data.mat');

training = data.training;
test = data.test;

nIter = 2000


label = training.labels;
image = training.images;
shape = size(image);



image = reshape(image,[shape(1)*shape(2), shape(3)]);
[data0 data1] = partition(image,label,0);

nA = size(data0,2);
nB = size(data1,2);
data0 = data0(:,1:800);
data1 = data1(:,1:4800);

R = 8;

x= zeros(shape(1)*shape(2)+1,1);
xOut = zeros(shape(1)*shape(2)+1,nIter);
%L = [1 2 5 10 20 50];
L = 20
objective = zeros(nIter,length(L));
gradOut = zeros(nIter,length(L));
for i = 1: length(L)
    [xOut gradOut(:,i)] = gradient1(x,nIter,R,L(i),data0,data1);
end
%%
figure
x = 1:nIter;
semilogy(x,gradOut(:,1),x,gradOut(:,2),x,gradOut(:,3),x,gradOut(:,4),x,gradOut(:,5),x,gradOut(:,6))
legend('L=1','L=2','L=5','L=10','L=20','L=50')

% 
% nA = size(data0);
% nB = size(data1);
% 

% for j = 1:nIter
%     
%     for i = 1:nA(2)
%         y1 = xOut(1:end-1,j)'*data0(:,i) +xOut(end,j);
%         objective(j,k) = objective(j,k) + log(1+exp(-y1))/nA(2);
%     end
% 
%     %accuracy21(j,k) = accuracy21(j,k)/nA(2) ;
%     
%     for i = 1:nB(2)
%         y2 = xOut(1:end-1,j)'*data1(:,i) + xOut(end,j);
%         objective(j,k) = objective(j,k) + log(1+exp(y2))/nB(2);
% 
%     end
%     %accuracy22(j,k) = accuracy22(j,k)/nB(2);
% end



