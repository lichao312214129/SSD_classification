% This script is used to regressing confounds: age, sex, headmotion and/or site.
% Before running this script, make sure you have run lc_demographic_information_statistics.py
% to get demographic information.

%================================================================
%% Correction of site, age, gender and head motion all together (Excluded subjects with greater head motion)
% Inputs
data_file = 'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_all.mat';
demographic_file = 'D:\WorkStation_2018\SZ_classification\Scale\demographic_all.xlsx';
% Load
data = importdata(data_file);
[demographic, header] = xlsread(demographic_file);
demographic(:,4) = demographic(:,4) == 1;

% Exclude subjects with greater head motion
loc_acceptable_headmotion = demographic(:,5)<=0.3;
demographic = demographic(loc_acceptable_headmotion,:);
data = data(loc_acceptable_headmotion,:);

% Regress 
site_design = zeros(size(demographic,1),4);
for i = 1:4
    site_design(:,i) = demographic(:,end) == i-1;
end

independent_variables_all = cat(2,site_design, demographic(:,[3 4 5]));
beta_value_all = independent_variables_all\data(:,3:end);
resid_all =  data(:,3:end) - independent_variables_all*beta_value_all;

% Get residual error
resid_all = cat(2, data(:,1:2),demographic(:,end), resid_all);
save('D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\fc_excluded_greater_fd_and_regressed_out_site_age_sex_motion_all.mat', 'resid_all');


%================================================================
%% Correction of age, gender and headmotion on training data and applied beta to test data (Exclude subjects with greater head motion)
% Inputs
data_file = 'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_all.mat';
demographic_file = 'D:\WorkStation_2018\SZ_classification\Scale\demographic_all.xlsx';
% Load
data = importdata(data_file);
[demographic, header] = xlsread(demographic_file);
demographic(:,4) = demographic(:,4) == 1;

% Exclude subjects with greater head motion
loc_acceptable_headmotion = demographic(:,5)<=0.3;
demographic = demographic(loc_acceptable_headmotion,:);
data = data(loc_acceptable_headmotion,:);
loc_train = demographic(:,end) ~=0;

% Fit sex and headmotion
indep_sex_headmotion_train = demographic(:,[3 4 5]);
indep_sex_headmotion_train = indep_sex_headmotion_train(loc_train, :);
dep_sex_headmotion_train = data(:,3:end);
dep_sex_headmotion_train = dep_sex_headmotion_train(loc_train, :);
beta_value_sex_headmotion_train = indep_sex_headmotion_train\dep_sex_headmotion_train;

% Regress out for all subjects
resid_all = data(:,3:end) - demographic(:,[3 4 5])*beta_value_sex_headmotion_train;

% Concat
resid_all = cat(2, data(:,1:2),demographic(:,end), resid_all);
save('D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\fc_excluded_greater_fd_and_regressed_out_age_sex_motion_separately.mat', 'resid_all');

%================================================================
%% For FEU: Correction of age, gender and headmotion together
% Inputs
data_file = 'D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\dataset_firstepisode_and_unmedicated_550.mat';
demographic_file = 'D:\WorkStation_2018\SZ_classification\Scale\demographic_all.xlsx';
% Load
data = importdata(data_file);
[demographic, header] = xlsread(demographic_file);
demographic(:,4) = demographic(:,4) == 1;

% Match demo and data
[h,loc] = ismember(data(:,1), demographic(:,1));
loc = loc(h);
demographic = demographic(loc, :);

% Exclude subjects with greater head motion
loc_acceptable_headmotion = demographic(:,5)<=0.3;
demographic = demographic(loc_acceptable_headmotion,:);
data = data(loc_acceptable_headmotion,:);

% Fit sex and headmotion
indep_sex_headmotion = demographic(:,[3 4 5]);
dep_sex_headmotion = data(:,3:end);
beta_value_sex_headmotion = indep_sex_headmotion\dep_sex_headmotion;

% Regress out for all subjects
resid_all = data(:,3:end) - demographic(:,[3 4 5])*beta_value_sex_headmotion;

% Concat
resid_all = cat(2, data(:,1:2),demographic(:,end), resid_all);
save('D:\WorkStation_2018\SZ_classification\Data\ML_data_npy\fc_feu_excluded_greater_fd_and_regressed_out_age_sex_motion_all.mat', 'resid_all');

%================================================================
%% Compare age, sex, headmotion between patients and hc
loc_p_site1 = (demographic(:,2)==1) & (demographic(:,6)==0);
loc_c_site1 = (demographic(:,2)==0) & (demographic(:,6)==0);
loc_p_site234 = (demographic(:,2)==1) & (demographic(:,6)~=0);
loc_c_site234 = (demographic(:,2)==0) & (demographic(:,6)~=0);

% Age
age = demographic(:,3);
age_p_site1 = age(loc_p_site1);
age_c_site1 = age(loc_c_site1);

age_p_site234 = age(loc_p_site234);
age_c_site234 = age(loc_c_site234);

[h,p,ci, tstat] = ttest2(age_p_site1, age_c_site1);
[h,p,ci, tstat] = ttest2(age_p_site234, age_c_site234);

save('D:\WorkStation_2018\SZ_classification\Scale\age_p_site1', 'age_p_site1');
save('D:\WorkStation_2018\SZ_classification\Scale\age_c_site1', 'age_c_site1');
save('D:\WorkStation_2018\SZ_classification\Scale\age_p_site234', 'age_p_site234');
save('D:\WorkStation_2018\SZ_classification\Scale\age_c_site234', 'age_c_site234');

% 
sex = demographic(:,4);
sex_site1 = sex(demographic(:,6)==0);
sex_p_site1 = sex(loc_p_site1);
sex_c_site1 = sex(loc_c_site1);
sex_site234 = sex(demographic(:,6)~=0);
sex_p_site234 = sex(loc_p_site234);
sex_c_site234 = sex(loc_c_site234);
sex_p_all = cat(1,sex_p_site1,sex_p_site234);sex_c_all = cat(1,sex_c_site1,sex_c_site234);

[p, Q]= chi2test([sumsex_p_site1==1), sumsex_p_site1==0);...
                        sumsex_c_site1==1), sumsex_c_site1==0)]);  % Site1

[p, Q]= chi2test([sumsex_p_site234==1), sumsex_p_site234==0);...
                        sumsex_c_site234==1), sumsex_c_site234==0)]); % Site 2 3 4 
                    
[p, Q]= chi2test([sumsex_p_all==1), sumsex_p_all==0);...
                        sumsex_c_all==1), sumsex_c_all==0)]); % Site 1 2 3 4 

[p, Q]= chi2test([sumsex_site1==1), sumsex_site1==0);...
                    sumsex_site234==1), sumsex_site234==0)]); % Site 1 2 3 4 
                

save('D:\WorkStation_2018\SZ_classification\Scale_sex_p_site1', 'sex_p_site1');
save('D:\WorkStation_2018\SZ_classification\Scale_sex_c_site1', 'sex_c_site1');
save('D:\WorkStation_2018\SZ_classification\Scale_sex_p_site234', 'sex_p_site234');
save('D:\WorkStation_2018\SZ_classification\Scale_sex_c_site234', 'sex_c_site234');

% Difference of head motion between patients and healthy controls of each site
loc_p_site1 = (demographic(:,2)==1) & (demographic(:,6)==0);
loc_p_site2 = (demographic(:,2)==1) & (demographic(:,6)==1);
loc_p_site3 = (demographic(:,2)==1) & (demographic(:,6)==2);
loc_p_site4 = (demographic(:,2)==1) & (demographic(:,6)==3);
loc_c_site1 = (demographic(:,2)==0) & (demographic(:,6)==0);
loc_c_site2 = (demographic(:,2)==0) & (demographic(:,6)==1);
loc_c_site3 = (demographic(:,2)==0) & (demographic(:,6)==2);
loc_c_site4 = (demographic(:,2)==0) & (demographic(:,6)==3);

headmotion=demographic(:,5);

headmotion_p_site1 = headmotion(loc_p_site1);
headmotion_p_site2 = headmotion(loc_p_site2);
headmotion_p_site3 = headmotion(loc_p_site3);
headmotion_p_site4 = headmotion(loc_p_site4);
headmotion_c_site1 = headmotion(loc_c_site1);
headmotion_c_site2 = headmotion(loc_c_site2);
headmotion_c_site3 = headmotion(loc_c_site3);
headmotion_c_site4 = headmotion(loc_c_site4);
headmotion_p_site234 = headmotion(loc_p_site234);
headmotion_c_site234 = headmotion(loc_c_site234);

[h,p,ci, tstat] = ttest2(headmotion_p_site1, headmotion_c_site1);
[h,p,ci, tstat] = ttest2(headmotion_p_site2, headmotion_c_site2);
[h,p,ci, tstat] = ttest2(headmotion_p_site3, headmotion_c_site3);
[h,p,ci, tstat] = ttest2(headmotion_p_site4, headmotion_c_site4);
[h,p,ci, tstat] = ttest2(headmotion_p_site234, headmotion_c_site234);

save('D:\WorkStation_2018\SZ_classification\Scale\headmotion_p_site1', 'headmotion_p_site1');
save('D:\WorkStation_2018\SZ_classification\Scale\headmotion_c_site1', 'headmotion_c_site1');
save('D:\WorkStation_2018\SZ_classification\Scale\headmotion_p_site234', 'headmotion_p_site234');
save('D:\WorkStation_2018\SZ_classification\Scale\headmotion_c_site234', 'headmotion_c_site234');


