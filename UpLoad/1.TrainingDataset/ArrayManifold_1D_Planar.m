function [b] = ArrayManifold_1D_Planar(LocationXY,f0,theta_in)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
n_signal = length(theta_in);
[n_array,numXY] = size(LocationXY);
b = zeros(n_array,n_signal);
c = 3e8;
theta = theta_in *pi/180;
for mm = 1:1:n_signal
    for kk = 1:1:n_array
        b(kk,mm) = exp(j*2*pi*f0/c* (LocationXY(kk,1)*sin(theta(mm))+LocationXY(kk,2)*cos(theta(mm))));
    end
end
end

