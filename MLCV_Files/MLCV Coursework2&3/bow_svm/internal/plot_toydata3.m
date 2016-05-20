function  plot_toydata3(data, z)

plot3(data(data(:,end)==1,1), data(data(:,end)==1,2),z(data(:,end)==1,1), 'o', 'MarkerFaceColor', [.9 .5 .5], 'MarkerEdgeColor','k');
hold on;
plot3(data(data(:,end)==2,1), data(data(:,end)==2,2),z(data(:,end)==2,1), 'o', 'MarkerFaceColor', [.5 .9 .5], 'MarkerEdgeColor','k');
hold on;
plot3(data(data(:,end)==3,1), data(data(:,end)==3,2),z(data(:,end)==3,1), 'o', 'MarkerFaceColor', [.5 .5 .9], 'MarkerEdgeColor','k');
hold on;
%axis([-1.5 1.5 -1.5 1.5]);
end

