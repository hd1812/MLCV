function  plot_toydataS( data )
plot(data(data(:,end)==1,1), data(data(:,end)==1,2), 's', 'MarkerFaceColor', [.9 .5 .5], 'MarkerEdgeColor','k','MarkerSize', 12);
hold on;
plot(data(data(:,end)==2,1), data(data(:,end)==2,2), 's', 'MarkerFaceColor', [.5 .9 .5], 'MarkerEdgeColor','k','MarkerSize', 12);
hold on;
plot(data(data(:,end)==3,1), data(data(:,end)==3,2), 's', 'MarkerFaceColor', [.5 .5 .9], 'MarkerEdgeColor','k','MarkerSize', 12);
hold on;
axis([-1.5 1.5 -1.5 1.5]);
end

