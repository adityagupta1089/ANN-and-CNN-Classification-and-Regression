close all
clear all
clc
for H = 2 .^ (1:6)
    for eta = 0.1 .^ (1:5)
        fprintf('H = %d, eta = %f\n', H, eta);
        l31(H, eta);
        close all
    end
end
