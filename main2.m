close all
clear all
clc

% load data
errors = {};
for i = 1:6 %H = 2 ^ i
    for j = 1:5 %eta = 0.1 ^ j
        fprintf('H = %d, eta = %f\n', 2 ^ i, 0.1 ^ j);
        errors{i,j} = load(sprintf('data/error_H%d_eta%f.mat', 2 ^ i, ...
            0.1 ^ j), 'trainerror');
        errors{i,j} = errors{i,j}.trainerror;
    end
end

colors = [[1 0 0]; [0 1 0]; [0 0 1]; [1 0 1]; [1 1 0]; [0 1 1]];

% effect of number of hidden layer nodes
% --------------------------------------
legends = {};
for i = 1:6
    legends{i} = sprintf('H = %d', 2 ^ i);
end

for j=1:5
    figure;
    hold on;
    for i = 1:6
        plot(errors{i,j}, 'Color', colors(i,:), 'LineWidth', 2);
    end
    legend(legends);
    xlabel('Epochs');
    ylabel('Sum of Squared Error');
    title(['\eta = ', num2str(0.1 ^ j)]);
    if j <= 2
        axes('position',[.2 .4 .45 .45])
        hold on;
        box on
        for i = 1:6
            plot(1:15, errors{i,j}(1:15), 'Color', colors(i,:), 'LineWidth', 2);
        end
        axis tight
    end
    if j <= 3
        axes('position',[.7 .2 .15 .3])
        hold on;
        box on
        for i = 1:6
            plot(900:1000, errors{i,j}(900:1000), 'Color', colors(i,:), 'LineWidth', 2);
        end
        axis tight
    end
    savefig(sprintf('fig2/eta%f.fig', 0.1 ^ j));
    saveas(gcf, sprintf('eps2/eta%f.eps', 0.1 ^ j), 'epsc');
end

% effect of learning rate
% -----------------------
legends = {};
for j = 1:5
    legends{j} = ['\eta = ', num2str(0.1 ^ j)];
end

for i=1:6
    figure;
    hold on;
    for j = 1:5
        plot(errors{i,j}, 'Color', colors(j,:), 'LineWidth', 2);
    end
    legend(legends);
    xlabel('Epochs');
    ylabel('Sum of Squared Error');
    title(['H = ', num2str(2 ^ i)]);
    savefig(sprintf('fig2/H%d.fig', 2 ^ i));
    saveas(gcf, sprintf('eps2/H%d.eps', 2 ^ i), 'epsc');
end
close all