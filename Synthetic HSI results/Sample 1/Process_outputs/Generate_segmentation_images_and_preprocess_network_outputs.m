%Here, network outputs (CAE + k-means, and end-to-end clustering) are 
% processed to produce segmentation images, and to enable calculation of 
% segmentation quality scores. 
%The code used to produce spectral k-means segmentation image is als 
% included but commented out, as the original input image 
% ('train_set_parsed.mat') could not be included in the repo. Though, 
% the repo does contain a previously generated
% segmentation image 'spec_kmeans.mat'

close all
clear
clc

%% Reconstruct end-to-end segmentation map
%load network output (cluster ID assigned to 3136 input patches)
load('cluster_out_train.mat')

%Convert labels into 20x20 patches
blank_patches = ones(3136,20,20,'single');%50000
cluster_ID_patches = squeeze(single(cluster_out_train(:))).*blank_patches;

%Reconstruct segmentation map
counter = 0;
for j = 20:4:240
    for i = 20:4:240
        counter= counter+1
        output(i-19:i,j-19:j) = cluster_ID_patches(counter,:,:);
    end
end
figure
imagesc(squeeze(output(:,:)))
set(gca,'Fontsize',18)
title('End-to-end clustering')
%Flatten to 1D array
output = output(:);
save('end_to_end.mat','output')

%% Ground truth segmentation map
%Load ground truth segmentation map
load('gt_end_to_end.mat')
figure
imagesc(squeeze(mat(:,:)))
set(gca,'Fontsize',18)
title('Ground truth')
%Flatten to 1D array
ground_truth = mat(:);
save('gt_image.mat','ground_truth')

%% Reconstruct CAE k-means segmentation image
clear output
% load latent vectors produced from pretrained CAE for all 3136 patches
load('encoded_imgs_pretrain.mat')
% cluster all latent vectors
idx = kmeans(encoded_imgs_pretrain,3);
%Convert labels into 20x20 patches
blank_patches = ones(3136,20,20,'single');%50000
cluster_ID_patches = squeeze(single(idx)).*blank_patches;

%Reconstruct segmentation image
counter = 0;
for j = 20:4:240
    for i = 20:4:240
        counter= counter+1
        output(i-19:i,j-19:j) = cluster_ID_patches(counter,:,:);
    end
end
figure
imagesc(squeeze(output(:,:)))
set(gca,'Fontsize',18)
title('CAE + k-means')
cae_kmeans = output;
%flatten into 1D array
cae_kmeans = cae_kmeans(:);
save('cae_kmeans.mat','cae_kmeans')

%% Spectral k-means segmentation image
%{
%
% NOTE: this section is commented out, as repo can't store train_set_parsed.mat
% HOWEVER, previously generated spec_kmeans.mat is included in the repo
%
%load HSI patches
load('train_set_parsed.mat')
%reconstruct HSI
clear output
counter = 0;
for j = 20:4:240
    for i = 20:4:240
        counter= counter+1
        output(i-19:i,j-19:j,:) = train_set_parsed(counter,:,:,:);
    end
end


%acquire all spectra from HSI
test_set_whole_reshaped = reshape(output,[],800);
%perform k-means clustering
idx = kmeans(double(test_set_whole_reshaped),3);
idx = reshape(idx,240,240);
figure
imagesc(squeeze(idx))
set(gca,'Fontsize',18)
title('Spectral k-means: 3 clusters')
%flatten into 1D array
spec_kmeans = idx;
spec_kmeans = spec_kmeans(:);

 save('spec_kmeans.mat','spec_kmeans')
%}