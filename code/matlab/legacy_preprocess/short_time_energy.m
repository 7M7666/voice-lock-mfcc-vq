function E = short_time_energy(frames)
%SHORT_TIME_ENERGY 帧短时能量 sum(x^2)


    E = sum(frames.^2, 2);   % 按行求和
end
