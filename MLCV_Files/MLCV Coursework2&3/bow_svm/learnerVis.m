function learnerVis(classificationID, m,data)

    if classificationID == 1
        var1=[-2:0.01:2]';
        var2=[-2:0.01:2]';
        if m.r ==1
            plot(m.t*ones(size(var1)),var1,'black');
        else
            plot(var1,m.t*ones(size(var1)),'black');
        end
        
    elseif classificationID ==2
    
        
    elseif classificationID ==3
               
        if m.r1==1&&m.r2==2
            
            var1=data(:,1);
            var2=data(:,2);
            [x, y] = meshgrid(var1,var2);
            z=m.w(1).*(var1.*var2)+m.w(2).*(var1.^2)+m.w(3).*(var2.^2)+m.w(4).*var1+m.w(5).*var2+m.w(6);
            
            plot_toydata3(data,z);
            x=-1.5:0.3:1.5;
            y=-1.5:0.1:1.5;
            n=size(x,3);
            mesh(x,y,ones(n,n)*m.t2)
            hold on
            if m.t1~=-inf
                mesh(x,y,ones(n,n)*m.t1)
            end
            hold on
            grid on;
            

        elseif m.r1==2&&m.r2==1
            var1=data(:,2);
            var2=data(:,1);
            [x, y] = meshgrid(var1,var2);
            z=m.w(1).*(var1.*var2)+m.w(2).*(var1.^2)+m.w(3).*(var2.^2)+m.w(4).*var1+m.w(5).*var2+m.w(6);
            plot_toydata3(data,z);
            
            x=-1.5:0.3:1.5;
            y=-1.5:0.3:1.5;
            n=size(x,2);
            mesh(x,y,ones(n,n)*m.t2)
            hold on
            if m.t1~=-inf
                mesh(x,y,ones(n,n)*m.t1)
            end
            grid on;
        end
        
        
    elseif classificationID ==4
        
    end
        
end