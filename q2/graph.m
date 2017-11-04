clc
clear all
close all
fID = fopen('part 4 new/learning rate 0.005/log_03_11_00_16.log', 'r');
A = textscan(fID, '%d Train:%f Test:%f', 'delimiter', '\t');
fclose(fID);
figure
hold on
plot(A{1}, A{2}, 'r', 'LineWidth', 2);
plot(A{1}, A{3}, 'b', 'LineWidth', 2);
xlabel('Epochs')
ylabel('Mean Sqaured Error')
legend('Training Error', 'Testing Error')