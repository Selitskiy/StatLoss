function [imageDS, dataSetFolders] = createBCbaselineE3(dataFolderTmpl, dataFolderSfx, readFcn)

% Create a real folder
dataFolder = strrep(dataFolderTmpl, 'Sfx', dataFolderSfx);


%% Create a vector of the baseline sets

% Empty vector
dataSetFolders = strings(0);
labels = strings(0);

% Let's populate the vector by the baseline folder templates, one by one
% 
dataSetFolders = [dataSetFolders, 'S1/S1NM1/S1NM1AN'];
%dataSetFolders = [dataSetFolders, 'S1/S1NM2/S1NM2AN'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,3);
labelCurN = strlength(labelCur);
labelCur = extractBetween(labelCur, labelCurN-1, labelCurN);
labels = [labels, labelCur];
%dataSetFolders = [dataSetFolders, 'S2/S2NM1/S2NM1AN'];
dataSetFolders = [dataSetFolders, 'S2/S2NM2/S2NM2AN'];

dataSetFolders = [dataSetFolders, 'S3/S3NM1/S3NM1AN'];
dataSetFolders = [dataSetFolders, 'S3/S3NM2/S3NM2AN'];
dataSetFolders = [dataSetFolders, 'S4/S4NM1/S4NM1AN'];
dataSetFolders = [dataSetFolders, 'S5/S5NM1/S5NM1AN'];
dataSetFolders = [dataSetFolders, 'S5/S5NM2/S5NM2AN'];
dataSetFolders = [dataSetFolders, 'S5/S5NM3/S5NM3AN'];
dataSetFolders = [dataSetFolders, 'S6/S6NM1/S6NM1AN'];
dataSetFolders = [dataSetFolders, 'S7/S7NM1/S7NM1AN'];
dataSetFolders = [dataSetFolders, 'S7/S7NM2/S7NM2AN'];
dataSetFolders = [dataSetFolders, 'S7/S7NM3/S7NM3AN'];
dataSetFolders = [dataSetFolders, 'S8/S8NM1/S8NM1AN'];
dataSetFolders = [dataSetFolders, 'S9/S9NM1/S9NM1AN'];
dataSetFolders = [dataSetFolders, 'S10/S10NM1/S10NM1AN'];
dataSetFolders = [dataSetFolders, 'S11/S11NM1/S11NM1AN'];
dataSetFolders = [dataSetFolders, 'S11/S11NM2/S11NM2AN'];
dataSetFolders = [dataSetFolders, 'S12/S12NM1/S12NM1AN'];
dataSetFolders = [dataSetFolders, 'S13/S13NM1/S13NM1AN'];
dataSetFolders = [dataSetFolders, 'S14/S14NM1/S14NM1AN'];
dataSetFolders = [dataSetFolders, 'S15/S15NM1/S15NM1AN'];
dataSetFolders = [dataSetFolders, 'S16/S16NM1/S16NM1AN'];
dataSetFolders = [dataSetFolders, 'S17/S17NM1/S17NM1AN'];
dataSetFolders = [dataSetFolders, 'S18/S18NM1/S18NM1AN'];
dataSetFolders = [dataSetFolders, 'S19/S19NM1/S19NM1AN'];
dataSetFolders = [dataSetFolders, 'S20/S20NM1/S20NM1AN'];
dataSetFolders = [dataSetFolders, 'S21/S21NM1/S21NM1AN'];
%

dataSetFolders = [dataSetFolders, 'S1/S1NM1/S1NM1CE'];
%dataSetFolders = [dataSetFolders, 'S1/S1NM2/S1NM2CE'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,3);
labelCurN = strlength(labelCur);
labelCur = extractBetween(labelCur, labelCurN-1, labelCurN);
labels = [labels, labelCur];
%dataSetFolders = [dataSetFolders, 'S2/S2NM1/S2NM1CE'];
dataSetFolders = [dataSetFolders, 'S2/S2NM2/S2NM2CE'];

dataSetFolders = [dataSetFolders, 'S3/S3NM1/S3NM1CE'];
dataSetFolders = [dataSetFolders, 'S3/S3NM2/S3NM2CE'];
dataSetFolders = [dataSetFolders, 'S4/S4NM1/S4NM1CE'];
dataSetFolders = [dataSetFolders, 'S5/S5NM1/S5NM1CE'];
dataSetFolders = [dataSetFolders, 'S5/S5NM2/S5NM2CE'];
dataSetFolders = [dataSetFolders, 'S5/S5NM3/S5NM3CE'];
dataSetFolders = [dataSetFolders, 'S6/S6NM1/S6NM1CE'];
dataSetFolders = [dataSetFolders, 'S7/S7NM1/S7NM1CE'];
dataSetFolders = [dataSetFolders, 'S7/S7NM2/S7NM2CE'];
dataSetFolders = [dataSetFolders, 'S7/S7NM3/S7NM3CE'];
dataSetFolders = [dataSetFolders, 'S8/S8NM1/S8NM1CE'];
dataSetFolders = [dataSetFolders, 'S9/S9NM1/S9NM1CE'];
dataSetFolders = [dataSetFolders, 'S10/S10NM1/S10NM1CE'];
dataSetFolders = [dataSetFolders, 'S11/S11NM1/S11NM1CE'];
dataSetFolders = [dataSetFolders, 'S11/S11NM2/S11NM2CE'];
dataSetFolders = [dataSetFolders, 'S12/S12NM1/S12NM1CE'];
dataSetFolders = [dataSetFolders, 'S13/S13NM1/S13NM1CE'];
dataSetFolders = [dataSetFolders, 'S14/S14NM1/S14NM1CE'];
dataSetFolders = [dataSetFolders, 'S15/S15NM1/S15NM1CE'];
dataSetFolders = [dataSetFolders, 'S16/S16NM1/S16NM1CE'];
dataSetFolders = [dataSetFolders, 'S17/S17NM1/S17NM1CE'];
dataSetFolders = [dataSetFolders, 'S18/S18NM1/S18NM1CE'];
dataSetFolders = [dataSetFolders, 'S19/S19NM1/S19NM1CE'];
dataSetFolders = [dataSetFolders, 'S20/S20NM1/S20NM1CE'];
dataSetFolders = [dataSetFolders, 'S21/S21NM1/S21NM1CE'];
%

dataSetFolders = [dataSetFolders, 'S1/S1NM1/S1NM1DS'];
%dataSetFolders = [dataSetFolders, 'S1/S1NM2/S1NM2DS'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,3);
labelCurN = strlength(labelCur);
labelCur = extractBetween(labelCur, labelCurN-1, labelCurN);
labels = [labels, labelCur];
%dataSetFolders = [dataSetFolders, 'S2/S2NM1/S2NM1DS'];
dataSetFolders = [dataSetFolders, 'S2/S2NM2/S2NM2DS'];

dataSetFolders = [dataSetFolders, 'S3/S3NM1/S3NM1DS'];
dataSetFolders = [dataSetFolders, 'S3/S3NM2/S3NM2DS'];
dataSetFolders = [dataSetFolders, 'S4/S4NM1/S4NM1DS'];
dataSetFolders = [dataSetFolders, 'S5/S5NM1/S5NM1DS'];
dataSetFolders = [dataSetFolders, 'S5/S5NM2/S5NM2DS'];
dataSetFolders = [dataSetFolders, 'S5/S5NM3/S5NM3DS'];
dataSetFolders = [dataSetFolders, 'S6/S6NM1/S6NM1DS'];
dataSetFolders = [dataSetFolders, 'S7/S7NM1/S7NM1DS'];
dataSetFolders = [dataSetFolders, 'S7/S7NM2/S7NM2DS'];
dataSetFolders = [dataSetFolders, 'S7/S7NM3/S7NM3DS'];
dataSetFolders = [dataSetFolders, 'S8/S8NM1/S8NM1DS'];
dataSetFolders = [dataSetFolders, 'S9/S9NM1/S9NM1DS'];
dataSetFolders = [dataSetFolders, 'S10/S10NM1/S10NM1DS'];
dataSetFolders = [dataSetFolders, 'S11/S11NM1/S11NM1DS'];
dataSetFolders = [dataSetFolders, 'S11/S11NM2/S11NM2DS'];
dataSetFolders = [dataSetFolders, 'S12/S12NM1/S12NM1DS'];
dataSetFolders = [dataSetFolders, 'S13/S13NM1/S13NM1DS'];
dataSetFolders = [dataSetFolders, 'S14/S14NM1/S14NM1DS'];
dataSetFolders = [dataSetFolders, 'S15/S15NM1/S15NM1DS'];
dataSetFolders = [dataSetFolders, 'S16/S16NM1/S16NM1DS'];
dataSetFolders = [dataSetFolders, 'S17/S17NM1/S17NM1DS'];
dataSetFolders = [dataSetFolders, 'S18/S18NM1/S18NM1DS'];
dataSetFolders = [dataSetFolders, 'S19/S19NM1/S19NM1DS'];
dataSetFolders = [dataSetFolders, 'S20/S20NM1/S20NM1DS'];
dataSetFolders = [dataSetFolders, 'S21/S21NM1/S21NM1DS'];
%

dataSetFolders = [dataSetFolders, 'S1/S1NM1/S1NM1HP'];
%dataSetFolders = [dataSetFolders, 'S1/S1NM2/S1NM2HP'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,3);
labelCurN = strlength(labelCur);
labelCur = extractBetween(labelCur, labelCurN-1, labelCurN);
labels = [labels, labelCur];
%dataSetFolders = [dataSetFolders, 'S2/S2NM1/S2NM1HP'];
dataSetFolders = [dataSetFolders, 'S2/S2NM2/S2NM2HP'];

dataSetFolders = [dataSetFolders, 'S3/S3NM1/S3NM1HP'];
dataSetFolders = [dataSetFolders, 'S3/S3NM2/S3NM2HP'];
dataSetFolders = [dataSetFolders, 'S4/S4NM1/S4NM1HP'];
dataSetFolders = [dataSetFolders, 'S5/S5NM1/S5NM1HP'];
dataSetFolders = [dataSetFolders, 'S5/S5NM2/S5NM2HP'];
dataSetFolders = [dataSetFolders, 'S5/S5NM3/S5NM3HP'];
dataSetFolders = [dataSetFolders, 'S6/S6NM1/S6NM1HP'];
dataSetFolders = [dataSetFolders, 'S7/S7NM1/S7NM1HP'];
dataSetFolders = [dataSetFolders, 'S7/S7NM2/S7NM2HP'];
dataSetFolders = [dataSetFolders, 'S7/S7NM3/S7NM3HP'];
dataSetFolders = [dataSetFolders, 'S8/S8NM1/S8NM1HP'];
dataSetFolders = [dataSetFolders, 'S9/S9NM1/S9NM1HP'];
dataSetFolders = [dataSetFolders, 'S10/S10NM1/S10NM1HP'];
dataSetFolders = [dataSetFolders, 'S11/S11NM1/S11NM1HP'];
dataSetFolders = [dataSetFolders, 'S11/S11NM2/S11NM2HP'];
dataSetFolders = [dataSetFolders, 'S12/S12NM1/S12NM1HP'];
dataSetFolders = [dataSetFolders, 'S13/S13NM1/S13NM1HP'];
dataSetFolders = [dataSetFolders, 'S14/S14NM1/S14NM1HP'];
dataSetFolders = [dataSetFolders, 'S15/S15NM1/S15NM1HP'];
dataSetFolders = [dataSetFolders, 'S16/S16NM1/S16NM1HP'];
dataSetFolders = [dataSetFolders, 'S17/S17NM1/S17NM1HP'];
dataSetFolders = [dataSetFolders, 'S18/S18NM1/S18NM1HP'];
dataSetFolders = [dataSetFolders, 'S19/S19NM1/S19NM1HP'];
dataSetFolders = [dataSetFolders, 'S20/S20NM1/S20NM1HP'];
dataSetFolders = [dataSetFolders, 'S21/S21NM1/S21NM1HP'];
%

dataSetFolders = [dataSetFolders, 'S1/S1NM1/S1NM1NE'];
%dataSetFolders = [dataSetFolders, 'S1/S1NM2/S1NM2NE'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,3);
labelCurN = strlength(labelCur);
labelCur = extractBetween(labelCur, labelCurN-1, labelCurN);
labels = [labels, labelCur];
%dataSetFolders = [dataSetFolders, 'S2/S2NM1/S2NM1NE'];
dataSetFolders = [dataSetFolders, 'S2/S2NM2/S2NM2NE'];

dataSetFolders = [dataSetFolders, 'S3/S3NM1/S3NM1NE'];
dataSetFolders = [dataSetFolders, 'S3/S3NM2/S3NM2NE'];
dataSetFolders = [dataSetFolders, 'S4/S4NM1/S4NM1NE'];
dataSetFolders = [dataSetFolders, 'S5/S5NM1/S5NM1NE'];
dataSetFolders = [dataSetFolders, 'S5/S5NM2/S5NM2NE'];
dataSetFolders = [dataSetFolders, 'S5/S5NM3/S5NM3NE'];
dataSetFolders = [dataSetFolders, 'S6/S6NM1/S6NM1NE'];
dataSetFolders = [dataSetFolders, 'S7/S7NM1/S7NM1NE'];
dataSetFolders = [dataSetFolders, 'S7/S7NM2/S7NM2NE'];
dataSetFolders = [dataSetFolders, 'S7/S7NM3/S7NM3NE'];
dataSetFolders = [dataSetFolders, 'S8/S8NM1/S8NM1NE'];
dataSetFolders = [dataSetFolders, 'S9/S9NM1/S9NM1NE'];
dataSetFolders = [dataSetFolders, 'S10/S10NM1/S10NM1NE'];
dataSetFolders = [dataSetFolders, 'S11/S11NM1/S11NM1NE'];
dataSetFolders = [dataSetFolders, 'S11/S11NM2/S11NM2NE'];
dataSetFolders = [dataSetFolders, 'S12/S12NM1/S12NM1NE'];
dataSetFolders = [dataSetFolders, 'S13/S13NM1/S13NM1NE'];
dataSetFolders = [dataSetFolders, 'S14/S14NM1/S14NM1NE'];
dataSetFolders = [dataSetFolders, 'S15/S15NM1/S15NM1NE'];
dataSetFolders = [dataSetFolders, 'S16/S16NM1/S16NM1NE'];
dataSetFolders = [dataSetFolders, 'S17/S17NM1/S17NM1NE'];
dataSetFolders = [dataSetFolders, 'S18/S18NM1/S18NM1NE'];
dataSetFolders = [dataSetFolders, 'S19/S19NM1/S19NM1NE'];
dataSetFolders = [dataSetFolders, 'S20/S20NM1/S20NM1NE'];
dataSetFolders = [dataSetFolders, 'S21/S21NM1/S21NM1NE'];
%

dataSetFolders = [dataSetFolders, 'S1/S1NM1/S1NM1SA'];
%dataSetFolders = [dataSetFolders, 'S1/S1NM2/S1NM2SA'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,3);
labelCurN = strlength(labelCur);
labelCur = extractBetween(labelCur, labelCurN-1, labelCurN);
labels = [labels, labelCur];
%dataSetFolders = [dataSetFolders, 'S2/S2NM1/S2NM1SA'];
dataSetFolders = [dataSetFolders, 'S2/S2NM2/S2NM2SA'];

dataSetFolders = [dataSetFolders, 'S3/S3NM1/S3NM1SA'];
dataSetFolders = [dataSetFolders, 'S3/S3NM2/S3NM2SA'];
dataSetFolders = [dataSetFolders, 'S4/S4NM1/S4NM1SA'];
dataSetFolders = [dataSetFolders, 'S5/S5NM1/S5NM1SA'];
dataSetFolders = [dataSetFolders, 'S5/S5NM2/S5NM2SA'];
dataSetFolders = [dataSetFolders, 'S5/S5NM3/S5NM3SA'];
dataSetFolders = [dataSetFolders, 'S6/S6NM1/S6NM1SA'];
dataSetFolders = [dataSetFolders, 'S7/S7NM1/S7NM1SA'];
dataSetFolders = [dataSetFolders, 'S7/S7NM2/S7NM2SA'];
dataSetFolders = [dataSetFolders, 'S7/S7NM3/S7NM3SA'];
dataSetFolders = [dataSetFolders, 'S8/S8NM1/S8NM1SA'];
dataSetFolders = [dataSetFolders, 'S9/S9NM1/S9NM1SA'];
dataSetFolders = [dataSetFolders, 'S10/S10NM1/S10NM1SA'];
dataSetFolders = [dataSetFolders, 'S11/S11NM1/S11NM1SA'];
dataSetFolders = [dataSetFolders, 'S11/S11NM2/S11NM2SA'];
dataSetFolders = [dataSetFolders, 'S12/S12NM1/S12NM1SA'];
dataSetFolders = [dataSetFolders, 'S13/S13NM1/S13NM1SA'];
dataSetFolders = [dataSetFolders, 'S14/S14NM1/S14NM1SA'];
dataSetFolders = [dataSetFolders, 'S15/S15NM1/S15NM1SA'];
dataSetFolders = [dataSetFolders, 'S16/S16NM1/S16NM1SA'];
dataSetFolders = [dataSetFolders, 'S17/S17NM1/S17NM1SA'];
dataSetFolders = [dataSetFolders, 'S18/S18NM1/S18NM1SA'];
dataSetFolders = [dataSetFolders, 'S19/S19NM1/S19NM1SA'];
dataSetFolders = [dataSetFolders, 'S20/S20NM1/S20NM1SA'];
dataSetFolders = [dataSetFolders, 'S21/S21NM1/S21NM1SA'];
%

dataSetFolders = [dataSetFolders, 'S1/S1NM1/S1NM1SC'];
%dataSetFolders = [dataSetFolders, 'S1/S1NM2/S1NM2SC'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,3);
labelCurN = strlength(labelCur);
labelCur = extractBetween(labelCur, labelCurN-1, labelCurN);
labels = [labels, labelCur];
%dataSetFolders = [dataSetFolders, 'S2/S2NM1/S2NM1SC'];
dataSetFolders = [dataSetFolders, 'S2/S2NM2/S2NM2SC'];

dataSetFolders = [dataSetFolders, 'S3/S3NM1/S3NM1SC'];
dataSetFolders = [dataSetFolders, 'S3/S3NM2/S3NM2SC'];
dataSetFolders = [dataSetFolders, 'S4/S4NM1/S4NM1SC'];
dataSetFolders = [dataSetFolders, 'S5/S5NM1/S5NM1SC'];
dataSetFolders = [dataSetFolders, 'S5/S5NM2/S5NM2SC'];
dataSetFolders = [dataSetFolders, 'S5/S5NM3/S5NM3SC'];
dataSetFolders = [dataSetFolders, 'S6/S6NM1/S6NM1SC'];
dataSetFolders = [dataSetFolders, 'S7/S7NM1/S7NM1SC'];
dataSetFolders = [dataSetFolders, 'S7/S7NM2/S7NM2SC'];
dataSetFolders = [dataSetFolders, 'S7/S7NM3/S7NM3SC'];
dataSetFolders = [dataSetFolders, 'S8/S8NM1/S8NM1SC'];
dataSetFolders = [dataSetFolders, 'S9/S9NM1/S9NM1SC'];
dataSetFolders = [dataSetFolders, 'S10/S10NM1/S10NM1SC'];
dataSetFolders = [dataSetFolders, 'S11/S11NM1/S11NM1SC'];
dataSetFolders = [dataSetFolders, 'S11/S11NM2/S11NM2SC'];
dataSetFolders = [dataSetFolders, 'S12/S12NM1/S12NM1SC'];
dataSetFolders = [dataSetFolders, 'S13/S13NM1/S13NM1SC'];
dataSetFolders = [dataSetFolders, 'S14/S14NM1/S14NM1SC'];
dataSetFolders = [dataSetFolders, 'S15/S15NM1/S15NM1SC'];
dataSetFolders = [dataSetFolders, 'S16/S16NM1/S16NM1SC'];
dataSetFolders = [dataSetFolders, 'S17/S17NM1/S17NM1SC'];
dataSetFolders = [dataSetFolders, 'S18/S18NM1/S18NM1SC'];
dataSetFolders = [dataSetFolders, 'S19/S19NM1/S19NM1SC'];
dataSetFolders = [dataSetFolders, 'S20/S20NM1/S20NM1SC'];
dataSetFolders = [dataSetFolders, 'S21/S21NM1/S21NM1SC'];
%

dataSetFolders = [dataSetFolders, 'S1/S1NM1/S1NM1SR'];
%dataSetFolders = [dataSetFolders, 'S1/S1NM2/S1NM2SR'];
[~, n] = size(dataSetFolders);
[tmpStr, ~] = strsplit(dataSetFolders(n), '/');
labelCur = tmpStr(1,3);
labelCurN = strlength(labelCur);
labelCur = extractBetween(labelCur, labelCurN-1, labelCurN);
labels = [labels, labelCur];
%dataSetFolders = [dataSetFolders, 'S2/S2NM1/S2NM1SR'];
dataSetFolders = [dataSetFolders, 'S2/S2NM2/S2NM2SR'];

dataSetFolders = [dataSetFolders, 'S3/S3NM1/S3NM1SR'];
dataSetFolders = [dataSetFolders, 'S3/S3NM2/S3NM2SR'];
dataSetFolders = [dataSetFolders, 'S4/S4NM1/S4NM1SR'];
dataSetFolders = [dataSetFolders, 'S5/S5NM1/S5NM1SR'];
dataSetFolders = [dataSetFolders, 'S5/S5NM2/S5NM2SR'];
dataSetFolders = [dataSetFolders, 'S5/S5NM3/S5NM3SR'];
dataSetFolders = [dataSetFolders, 'S6/S6NM1/S6NM1SR'];
dataSetFolders = [dataSetFolders, 'S7/S7NM1/S7NM1SR'];
dataSetFolders = [dataSetFolders, 'S7/S7NM2/S7NM2SR'];
dataSetFolders = [dataSetFolders, 'S7/S7NM3/S7NM3SR'];
dataSetFolders = [dataSetFolders, 'S8/S8NM1/S8NM1SR'];
dataSetFolders = [dataSetFolders, 'S9/S9NM1/S9NM1SR'];
dataSetFolders = [dataSetFolders, 'S10/S10NM1/S10NM1SR'];
dataSetFolders = [dataSetFolders, 'S11/S11NM1/S11NM1SR'];
dataSetFolders = [dataSetFolders, 'S11/S11NM2/S11NM2SR'];
dataSetFolders = [dataSetFolders, 'S12/S12NM1/S12NM1SR'];
dataSetFolders = [dataSetFolders, 'S13/S13NM1/S13NM1SR'];
dataSetFolders = [dataSetFolders, 'S14/S14NM1/S14NM1SR'];
dataSetFolders = [dataSetFolders, 'S15/S15NM1/S15NM1SR'];
dataSetFolders = [dataSetFolders, 'S16/S16NM1/S16NM1SR'];
dataSetFolders = [dataSetFolders, 'S17/S17NM1/S17NM1SR'];
dataSetFolders = [dataSetFolders, 'S18/S18NM1/S18NM1SR'];
dataSetFolders = [dataSetFolders, 'S19/S19NM1/S19NM1SR'];
dataSetFolders = [dataSetFolders, 'S20/S20NM1/S20NM1SR'];
dataSetFolders = [dataSetFolders, 'S21/S21NM1/S21NM1SR'];
%


%% Replace Sfx template with the actual value
dataSetFolders = strrep(dataSetFolders, 'Sfx', dataFolderSfx);
labels = strrep(labels, 'Sfx', dataFolderSfx);

% Build a full path  
fullDataSetFolders = fullfile(dataFolder, dataSetFolders);


[~, m] = size(labels);


%% Create a Datastore of the baseline images with individual folder lables
imageDS = imageDatastore(fullDataSetFolders, 'IncludeSubfolders', false,...
                            'LabelSource', 'none');                        
imageDS.ReadFcn = readFcn;


%% Label images by top subject's folder name                        
[n, ~] = size(imageDS.Files);  
labelStr = strings(n,1);

fprintf("Labeling base set:\n");

for j=1:m
    
    fprintf(" %s,", labels(j));
     
    matches = contains(string(imageDS.Files), strcat(labels(j),'/'));
    labelStr(matches) = labels(j); 
    
end
fprintf("\n");

imageDS.Labels = categorical(labelStr);

end

