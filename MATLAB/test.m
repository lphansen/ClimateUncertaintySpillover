clear
lat= [ 35.47 35.51 35.54 35.58 35.61 49.23 49.25 49.26 49.28];
lon= [ 48.91 48.95 49.00 49.04 49.09 72.77 72.84 72.91 72.97];
tol = 1; % distance > tol indicates discontinuity
dl = diff([lat;lon],1,2); % look up what that command does if you don't know it
euler_dist = sqrt((dl(1,:)+dl(2,:)).^2); % distance between data points
jumpind = [0 euler_dist>tol]; % now if jumpind(i) = true, we know that the 
                 %   point [lat(i) lon(i)] is the first after a jump
blocks = cumsum(jumpind); % points that belong to the same continuous part
                           % have the same value in blocks
% Now just loop over the continuous blocks to draw a separate line for each one
for i=0:blocks(end)
    plot(lat(blocks==i),lon(blocks==i),'LineWidth',2);
    hold on;
end
