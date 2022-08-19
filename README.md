# StatLoss
Code for family of papers that explored meta-learning Supervisor Neural Network with learnable Trustworthy Threshold in:
1. Static training mode LOD2022
2. Continous SNN test-training mode AIVR2022
3. Active underlying CNN learning using SNN uncertainty hints AGI2022

List of files:

 - Program files to re-train Inception v.3 (found the best out of 6 SOTA models) CNN model and train SNN:
Inception3BC2msrACL.m - Face Recognition (FR) task with learnable Trustworthy Threshold using Statistical Memory Loss function
Inception3BC2EmsrACL.m - Face Expression Recognition (FER) -"-

To configure, find and modify the following fragments:

FER specific:
  %% CONFIGURATION PARAMETERS:
  % Download BookClub dataset from: https://drive.google.com/file/d/1_U8SypDurlHV4c8NvBvdATn9g6SShPjs/view?usp=sharing
  % and unarchive it into the dierctory below:
  %% Dataset root folder template and suffix
dataFolderTmpl = '~/data/BC2E_Sfx';
dataFolderSfx = '1072x712';
  %Set directory and template for the retrained CNN models:
save_net_fileT = '~/data/in_eswarm';

FR specific:
  % Download BookClub dataset from: https://data.mendeley.com/datasets/yfx9h649wz/3
  % and unarchive it into the dierctory below:
  %% Dataset root folder template and suffix
dataFolderTmpl = '~/data/BC2_Sfx';
dataFolderSfx = '1072x712';
  %Set directory and template for the retrained CNN models:
save_net_fileT = '~/data/in_swarm';
  
Generic:
  %Set number of models in the ensemble:
nModels = 7;
  %Continous/Online/Lifetime learning mode 
contF = 1;
  %Structured test data
structF = 1;
  %Percentage of the allowed Oracle request (ACtive learning)
nOrcaleLimit = 0.001;

 - Libraries:
   * SNN custom layers
fullyConnectedM1ReluLayer.m
fullyConnectedCLLayer.m
TCLmRegression.m

   * Uncertainty Shape descriptor
makeUSDstrong.m

   * Ensemble voting with SNN verdict
ensemblePredictedLabels.m
   
   * Training and test sets building
createBCbaselineIDS6b1(...5).m
createBCtestIDSvect6b1(...5).m
createBCtestIDSvect6b.m
createBCtestIDSvect6bWhole.m
createBCbaselineE1(...5).m
createBCtestE1(...5).m
createBCtestE.m
createBCtestEWhole.m

   * Image size conversion:
readFunctionTrainIN_n.m

 - Accuracy metrics calculation script:
pred_dist2emsrCL_ol.R
pred_dist2emsrCL_kfold.R
