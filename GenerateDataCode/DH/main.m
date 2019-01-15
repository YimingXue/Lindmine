garbage_crop_2=zeros(2205,3000,63);
garbage_crop_4=zeros(1853,3000,63);
garbage_crop_15=zeros(1435,3000,63);
garbage_crop_23=zeros(1677,3000,63);
garbage_crop_27=zeros(1275,1925,63);
garbage_crop_37=zeros(561,591,63);
garbage_crop_38=zeros(635,663,63);
garbage_crop_40=zeros(1417,1807,63);
garbage_crop_43=zeros(1517,938,63);
garbage_crop_60=zeros(701,957,63);
garbage_crop_61=zeros(758,1098,63);
garbage_crop_75=zeros(957,1493,63);
%crop_59=zeros(2100,2840,63);
for i=1:63;
    garbage_crop_2(:,:,i)=multibandread('/media/xueyiming/新加卷/P2A_20180624132708383_0126_VNIR.dat',[7600,6000,63],'uint16=>uint16',0,'bsq','ieee-le',{'Row', 'Range', [5325 7529]},{'Column', 'Range', [2801 5800]},{'Band', 'Direct',i});
    garbage_crop_4(:,:,i)=multibandread('/media/xueyiming/新加卷/P2A_20180624132708383_0126_VNIR.dat',[4700,11500,63],'uint16=>uint16',0,'bsq','ieee-le',{'Row', 'Range', [2841 4693]},{'Column', 'Range', [8401 11400]},{'Band', 'Direct',i});
    garbage_crop_15(:,:,i)=multibandread('/media/xueyiming/新加卷/P2A_20180624132708383_0126_VNIR.dat',[4000,42300,63],'uint16=>uint16',0,'bsq','ieee-le',{'Row', 'Range', [2479 3913]},{'Column', 'Range', [39201 42200]},{'Band', 'Direct',i});
    garbage_crop_23(:,:,i)=multibandread('/media/xueyiming/新加卷/P2A_20180624132708383_0126_VNIR.dat',[5500,64700,63],'uint16=>uint16',0,'bsq','ieee-le',{'Row', 'Range', [3757 5433]},{'Column', 'Range', [61601 64600]},{'Band', 'Direct',i});
    garbage_crop_27(:,:,i)=multibandread('/media/xueyiming/新加卷/P2A_20180624132708383_0126_VNIR.dat',[8000,75400,63],'uint16=>uint16',0,'bsq','ieee-le',{'Row', 'Range', [6709 7983]},{'Column', 'Range', [73413 75337]},{'Band', 'Direct',i});
    garbage_crop_37(:,:,i)=multibandread('/media/xueyiming/新加卷/P2A_20180624132708383_0126_VNIR.dat',[8600,101600,63],'uint16=>uint16',0,'bsq','ieee-le',{'Row', 'Range', [8023 8583]},{'Column', 'Range', [100943 101533]},{'Band', 'Direct',i});
    garbage_crop_38(:,:,i)=multibandread('/media/xueyiming/新加卷/P2A_20180624132708383_0126_VNIR.dat',[8400,104300,63],'uint16=>uint16',0,'bsq','ieee-le',{'Row', 'Range', [7727 8361]},{'Column', 'Range', [103605 104267]},{'Band', 'Direct',i});
    garbage_crop_40(:,:,i)=multibandread('/media/xueyiming/新加卷/P2A_20180624132708383_0126_VNIR.dat',[4500,111200,63],'uint16=>uint16',0,'bsq','ieee-le',{'Row', 'Range', [3019 4435]},{'Column', 'Range', [109329 111135]},{'Band', 'Direct',i});
    garbage_crop_43(:,:,i)=multibandread('/media/xueyiming/新加卷/P2A_20180624132708383_0126_VNIR.dat',[6500,120700,63],'uint16=>uint16',0,'bsq','ieee-le',{'Row', 'Range', [4887 6403]},{'Column', 'Range', [119663 120600]},{'Band', 'Direct',i});
    garbage_crop_60(:,:,i)=multibandread('/media/xueyiming/新加卷/P2A_20180624132708383_0126_VNIR.dat',[8500,167900,63],'uint16=>uint16',0,'bsq','ieee-le',{'Row', 'Range', [7747 8447]},{'Column', 'Range', [166861 167817]},{'Band', 'Direct',i});
    garbage_crop_61(:,:,i)=multibandread('/media/xueyiming/新加卷/P2A_20180624132708383_0126_VNIR.dat',[4500,170700,63],'uint16=>uint16',0,'bsq','ieee-le',{'Row', 'Range', [3696 4453]},{'Column', 'Range', [169535 170632]},{'Band', 'Direct',i});
    garbage_crop_75(:,:,i)=multibandread('/media/xueyiming/新加卷/P2A_20180624132708383_0126_VNIR.dat',[5200,209700,63],'uint16=>uint16',0,'bsq','ieee-le',{'Row', 'Range', [4193 5149]},{'Column', 'Range', [208171 209663]},{'Band', 'Direct',i});
    %crop_59(:,:,i)=multibandread('/media/xueyiming/新加卷/P2A_20180624132708383_0126_VNIR.dat',[8200,165500,63],'uint16=>uint16',0,'bsq','ieee-le',{'Row', 'Range', [6000 8099]},{'Column', 'Range', [162560 165399]},{'Band', 'Direct',i});
    i
end
save('garbage_crop_2.mat','garbage_crop_2','-v7.3')
save('garbage_crop_4.mat','garbage_crop_4','-v7.3')
save('garbage_crop_15.mat','garbage_crop_15','-v7.3')
save('garbage_crop_23.mat','garbage_crop_23','-v7.3')
save('garbage_crop_27.mat','garbage_crop_27')
save('garbage_crop_37.mat','garbage_crop_37')
save('garbage_crop_38.mat','garbage_crop_38')
save('garbage_crop_40.mat','garbage_crop_40')
save('garbage_crop_43.mat','garbage_crop_43')
save('garbage_crop_60.mat','garbage_crop_60')
save('garbage_crop_61.mat','garbage_crop_61')
save('garbage_crop_75.mat','garbage_crop_75')

