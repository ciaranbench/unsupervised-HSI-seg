%Here, the end-to-end network output (end-to-end clustering) is 
% processed to produce segmentation images, and to enable calculation of 
% segmentation quality scores. 
%The code used to produce spectral k-means segmentation image is als 
% included but commented out, as the original input image 
% ('train_set_parsed.mat') could not be included in the repo. Though, 
% the repo does contain a previously generated
% segmentation image 'spec_kmeans.mat'

clear
close all
clc

%% Ground truth image
imageData_gt = load('Indian_pines_gt.mat');
imageData_gt = imageData_gt.indian_pines_gt;
%Image is cropped to compensate for tiling offset of segmentation image
imageData_gt = imageData_gt(3:140,3:140);

figure
imagesc(squeeze(imageData_gt))
set(gca,'Fontsize',18)
title('Ground truth')
imageData_gt = imageData_gt(:);
%Remove array elements corresponding to the 'background' of the image
zero_coordinates = find(imageData_gt == 0);
imageData_gt(zero_coordinates) = [];
save('gt.mat','imageData_gt')

%% CAE+k-means

load('encoded_imgs_pretrain.mat')
idx = kmeans(encoded_imgs_pretrain,17);
%Convert labels into 10x10 patches
blank_patches = ones(18496,10,10,'single');
cluster_ID_patches = squeeze(single(idx)).*blank_patches;

%reconstruct segmentation image
output = ones(280,280,'single');
%clear cluster_ID_patches
counter = 0;
for j = 10:2:280
    for i = 10:2:280
        counter= counter+1
        output(i-9:i,j-9:j) = squeeze(cluster_ID_patches(counter,:,:));
    end
end
cae_kmeans = output;
%downsample image to match dimensions of ground truth
cae_kmeans = transpose(downsample(transpose(downsample(cae_kmeans,2)),2));
cae_kmeans = cae_kmeans(1:138,1:138);
figure
imagesc(squeeze(cae_kmeans))
set(gca,'Fontsize',18)
title('CAE + k-means')
cae_kmeans = cae_kmeans(:);
%again, remove coordinates that correspond to the background. 
cae_kmeans(zero_coordinates) = [];
save('cae_kmeans.mat','cae_kmeans')


%% spectral k-means
%{
imageData = load('Indian_pines_corrected.mat');
imageData = imageData.indian_pines_corrected;
imageData = imageData(1:140,1:140,:);
test_set_whole_reshaped = reshape(imageData,[],200);
idx = kmeans(double(test_set_whole_reshaped),17);
idx = reshape(idx,140,140);

idx = idx(3:140,3:140);
figure
imagesc(squeeze(idx))
set(gca,'Fontsize',18)
title('Spectral k-means: 3 clusters')
idx = idx(:);
idx(zero_coordinates) = [];
save('kmeans.mat','idx')
%}




%% end-to-end clustering

load('cluster_out_train.mat')
%Convert labels into 20x20 patches
blank_patches = ones(18496,10,10,'single');
cluster_ID_patches = squeeze(single(cluster_out_train(:))).*blank_patches;
%reconstruct segmentation image
output = ones(280,280,'single');
%clear cluster_ID_patches
counter = 0;
for j = 10:2:280
    for i = 10:2:280
        counter= counter+1
        output(i-9:i,j-9:j) = squeeze(cluster_ID_patches(counter,:,:));
    end
end
%downsample to match dimensions of ground truth
output = transpose(downsample(transpose(downsample(output,2)),2));
output = output(1:138,1:138);

figure
imagesc(squeeze(output))
set(gca,'Fontsize',18)
title('End-to-end clustering')
output = output(:);
output(zero_coordinates) = [];
save('end_to_end.mat','output')









