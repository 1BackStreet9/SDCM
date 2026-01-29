close all; clc; clear;
pco = importdata("original_pc.txt");
pcm = importdata("simden_pc.txt");

shows(pco,1);
shows(pcm,2);

spco = VoxelHeat(pco, 1, 3);
s = pcolor(spco);
s.EdgeColor=[0,0,0];
% vod 根据自己的范围设置
xlim([10 50]);
ylim([5 45]);
caxis([0 20]);
colormap jet;
colorbar; 

spcm = VoxelHeat(pcm, 1, 4);
sm = pcolor(spcm);
sm.EdgeColor=[0,0,0];
sm.LineWidth = 0.5;
% vod
xlim([10 50]);
ylim([5 45]);
caxis([0 80]);
axis on;
colormap jet;
colorbar; 
function shows(point_cloud, n)
    figure1 = figure(n);
    axes1 = axes('Parent',figure1);
    scatter3(point_cloud(:,1), point_cloud(:,2), point_cloud(:,3), 5, point_cloud(:,3), 'filled');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    
    %%%%%%%%%%%% VoD%%%%%%%%%%%%%%%%%%
    %根据自己的范围设置
   %xlim(axes1,[5.54254965245496 47.953189546321]);
   %ylim(axes1,[-16.4464627061584 13.2403705799156]);
   %zlim(axes1,[-4.02737113189079 4.39754474574975]);
   %view(axes1,[-71.0999999811248 60.5999992409377]);

   set(axes1,'GridAlpha',0.4);
end

function s = VoxelHeat(points,voxel_size,n)
num_points = size(points,1);

x_indices = floor(points(:, 1) / voxel_size);  
y_indices = floor(points(:, 2) / voxel_size);  

min_x = min(x_indices);
max_x = max(x_indices);
min_y = min(y_indices);
max_y = max(y_indices);

grid_width = max_x - min_x + 1;
grid_height = max_y - min_y + 1;


grid = zeros(grid_width, grid_height);


for i = 1:num_points

    x_idx = x_indices(i) - min_x + 1; 
    y_idx = y_indices(i) - min_y + 1;  
    
    grid(x_idx, y_idx) = grid(x_idx, y_idx) + 1;
end

figure(n);
s = fliplr(grid);

end