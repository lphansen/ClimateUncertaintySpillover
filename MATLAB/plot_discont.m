function plot_discont(x, y)
tol = 1;
dl = diff([x;y],1,2);
euler_dist = sqrt( dl(1,:).^2 + dl(2,:).^2 );
jumpind = [0 euler_dist>tol]; 
blocks = cumsum(jumpind);
for i=0:blocks(end)
    k=sum(x(blocks==i));
    if k ==1
        plot(x(blocks==i),y(blocks==i),'*');
        hold on;
    else
        plot(x(blocks==i),y(blocks==i),'LineWidth',4);
        hold on;
    end
end
hold off
end

