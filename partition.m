function [data0,data1] = partition(im,label,k)

shape = size(im);
data0 = [];
data1 = [];

for i = 1:shape(2)
   if label(i) == k
       data0 = [data0 im(:,i)];
   else
       data1 = [data1 im(:,i)];
   end
end



end