%%
% This code is to extract ROI-wise brain signals from imaging data, and calculate the static functional connectivity between them

% Input: sub-xx_task-alice_bold_preprocessed.nii.gz
% Output: ROIsignals/ Static functional connectivity

% any question about the code, contact shuer.ye@ntnu
%%
% Run the code under Alice folder (as work path)
% The downloaded data (everything under the derivatives) should be put in the Subjects Folder

%%

% load parcellations (brain atlas), 400 Regions of interests (ROI) that can
% be assigned into 7 distinct brain networks
atlas=niftiread('Schaefer2018_400Parcels_7Networks_order_FSLMNI152_3mm.nii');

% details of this parcel can be found in 
% https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/fsleyes_lut/Schaefer2018_400Parcels_7Networks_order.lut

path1=[pwd,'/Subjects/'];
subs=dir([path1,'sub*']);

ROI=max(reshape(atlas,[],1));
out_path=[path1,'Results/'];

%%

for f1=1:length(subs)
    % Skip any subject before sub-39
    if str2double(subs(f1).name(5:end)) < 39
        continue; % Skip to the next subject
    end
    
    tic
    % Skip folder "sub-38" specifically
    if strcmp(subs(f1).name, 'sub-38')
        disp('Skipping sub-38 as the file does not exist');
        continue; % Skip this iteration
    end
     
    disp(['working on ',subs(f1).name])

    img_path=dir([path1,subs(f1).name,'/*preprocessed.nii']);

    % if its nii.gz, unzip it
    if isempty(img_path)
        zips=dir([path1,subs(f1).name,'/*preprocessed.nii.gz']);
        if ~isempty(zips)
            gunzip([path1,subs(f1).name,'/',zips(1).name]); % assume each subject only has one nii.gz file
        else
            error(['No preprocessed .nii.gz file found for ', subs(f1).name]);
        end
    end 

    % load subject data
    img_path=dir([path1,subs(f1).name,'/*preprocessed.nii']);
    ds_img=niftiread([path1,subs(f1).name,'/',img_path(1).name]);

    % extract ROI signals
    for tp=1:size(ds_img,4)   % TP = time point
        img_tmp=squeeze(ds_img(:,:,:,tp));
        for i=1:ROI
            img_tmp2=img_tmp(atlas==i);
            img_tmp2(img_tmp2==0)=NaN;
            ROISignal(i,tp)=mean(img_tmp2,'omitnan');
        end
    end

    % Corrected file paths with f1 instead of i
    filename1=[out_path,'ROIsignal/',subs(f1).name,'_ROISignal.mat'];
    filename2=[out_path,'FCmat/',subs(f1).name,'_FCmat.mat'];
    filename3=[out_path,'Pvalue/',subs(f1).name,'_Pvalue.mat'];
    filename4=[out_path,'zFCmat/',subs(f1).name,'_zFCmat.mat'];
    
    % Create necessary output directories
    mkdir([out_path,'ROIsignal/']);
    mkdir([out_path,'FCmat/']);
    mkdir([out_path,'Pvalue/']);
    mkdir([out_path,'zFCmat/']);


    % build functional connectivity matrix
    [fcmatrix,pmatrix]=corrcoef(ROISignal');
    zfcmatrix=atanh(fcmatrix);

    % save output
    save(filename1,"ROISignal");
    save(filename2,"fcmatrix");
    save(filename3,"pmatrix");
    save(filename4,"zfcmatrix");
    
    toc
end
