function LBPHist=LBPFunc(img)
    [row,col]=size(img);
    binMat=[1,2,4;128,0,8;64,32,16];
    mat=zeros(3,3);
    LBPHist=zeros(1,256);
    for i=2:row-1
        for j=2:col-1
            mat=repmat(img(i,j),3,3)>=img(i-1:i+1,j-1:j+1);
            lbpCode=sum(sum(mat.*binMat));
            LBPHist(lbpCode+1)=LBPHist(lbpCode+1)+1;
        end
    end
    %LBPHist=LBPHist/sum(LBPHist);
    minLBPHist=min(LBPHist);
    maxLBPHist=max(LBPHist);
    LBPHist=(LBPHist-minLBPHist)/(maxLBPHist-minLBPHist);
end