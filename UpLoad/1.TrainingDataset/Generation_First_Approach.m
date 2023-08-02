clc
clear all
close all
%%========参数设置============================================================
c = 3e8; 
f0 = 1e9; 
mda = c/f0;
Num_time_varying = 2;
Num_array = 10;
d_array1 = 0.6 * mda;
d_array2 = 0.9 * mda;
LocationXY_1st = zeros(Num_array, 2);
LocationXY_2nd = zeros(Num_array, 2);
for hh = 1:1:Num_array
    LocationXY_1st(hh,:) = [(hh - 1) * d_array1, 0];
    LocationXY_2nd(hh,:) = [(hh - 1) * d_array2, 0];
end
LocationXY = zeros(Num_array, 2, Num_time_varying);
LocationXY(:,:,1) = LocationXY_1st;
LocationXY(:,:,2) = LocationXY_2nd;
%%-----------------生成所有的DOA组数
Grid = 1;
DOA_begin = -90;
DOA_end = 90;
DOA_grid = DOA_begin:Grid:DOA_end;
%取遍网格所有可能
Num_True_DOA = 2;
DOA_all = [];
for ii=1:1:length(DOA_grid)
    disp(ii)
    for jj=ii+1:1:length(DOA_grid)
        DOA_all = [DOA_all ; DOA_begin+(ii-1)*Grid, DOA_begin+(jj-1)*Grid]; 
    end
end
%%===========生成Labels==============================================================
Num_TrainSample = size(DOA_all,1); %训练样本数
Train_output = zeros(Num_TrainSample, length(DOA_grid));
DOA_all_index = (DOA_all-DOA_begin)/Grid + 1;
DOA_all_index = int64(DOA_all_index);
for kk = 1:1:Num_TrainSample
    for mm = 1:1:Num_True_DOA
        Train_output(kk, DOA_all_index(kk,mm)) = 1;
    end
end
%%===========生成Inputs==============================================================
A_all_grid = [];
for nn = 1:1:Num_time_varying
    for pp = 1:1:length(DOA_grid)
        A_all_grid(:,pp,nn) = ArrayManifold_1D_Planar(LocationXY(:,:,nn),f0,DOA_grid(pp));
    end
end

R_a = [];
for qq = 1:1:Num_time_varying
    for rr = 1:1:Num_array
        for ss = rr+1:1:Num_array
            R_a_row = A_all_grid(rr,:,qq) .* conj(A_all_grid(ss,:,qq));
            R_a = [R_a ; R_a_row];
        end
    end
end

Train_input = zeros(Num_TrainSample,3,length(DOA_grid));
for tt = 1:1:Num_TrainSample
    disp(tt)
    R_shape = [];
    R = zeros(Num_array, Num_array, Num_time_varying);
    for uu = 1:1:Num_time_varying
        A_DOA = ArrayManifold_1D_Planar(LocationXY(:,:,uu),f0,DOA_all(tt,:));
        R(:,:,uu) = A_DOA * eye(Num_True_DOA) * A_DOA';
    end
    for vv = 1:1:Num_time_varying
        for ww = 1:1:Num_array
            for xx = ww+1:1:Num_array
                R_shape = [R_shape;R(ww,xx,vv)];
            end
        end
    end
    Spectrum_grid = R_a'*R_shape;
    Train_input(tt,1,:) = real(Spectrum_grid);
    Train_input(tt,2,:) = imag(Spectrum_grid);
    Train_input(tt,3,:) = angle(Spectrum_grid);
end

Train_input = single(Train_input);
Train_output = single(Train_output);
save('Dataset_First_Approach.mat','Train_input','Train_output')


