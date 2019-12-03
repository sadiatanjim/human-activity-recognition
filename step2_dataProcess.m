clc, clear all, close all

FEATSIZE = 66;

load('frames.mat');

sz = size(frames);
NFRAMES = sz(1);
FRAMESIZE = sz(2);

fs = 20;

feat = zeros(NFRAMES,FEATSIZE);

for i = 1:NFRAMES
    frame = frames(1,:,:);
    atx = frame(1,:,1);
    aty = frame(1,:,2);
    atz = frame(1,:,3);
    feat(i,:) = featuresFromBuffer(atx,aty,atz,fs);
end

save('features.mat','feat');