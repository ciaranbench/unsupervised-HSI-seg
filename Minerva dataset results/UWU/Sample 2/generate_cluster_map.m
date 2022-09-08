clear

%load network output (cluster ID assigned to 13456 input patches)
load('cluster_out_train.mat')

%Convert labels into 10x10 patches
blank_patches = ones(13456,10,10,'single');%50000
cluster_ID_patches = squeeze(single(cluster_out_train(:))).*blank_patches;

%Reconstruct segmentation map
counter = 0;
for j = 10:2:240
    for i = 10:2:240
        counter= counter+1
        output(i-9:i,j-9:j) = cluster_ID_patches(counter,:,:);
    end
end
figure
imagesc(squeeze(output(:,:)))
set(gca,'Fontsize',18);
