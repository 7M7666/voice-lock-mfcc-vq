
rawRoot   = '../LibriSpeech/dev-clean';              
cleanRoot = '../LibriSpeech_clean_mat/dev-clean';    

saveFrames = false;    
saveFig    = true;    
maxFigs   = 3;      
figStride = 100;      

preprocess_folder(rawRoot, cleanRoot, saveFrames, saveFig, maxFigs, figStride);
