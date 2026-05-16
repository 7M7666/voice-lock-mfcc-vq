appDir = fileparts(mfilename('fullpath'));
projectRoot = fullfile(appDir, '..', '..', '..');
cd(projectRoot);
addpath(appDir);
app = SRDPVoiceLockAppFixed;
assignin('base', 'SRDPVoiceLockAppFixedInstance', app);
