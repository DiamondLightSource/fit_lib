import numpy as np
import matplotlib.pyplot as plt
import time
import functions_for_beam_analyse as fs
import numpy.fft as F


#import cProfile, pstats, StringIO


# plot the allan deviation of the params of images 
  
def ald_plots(V,fps,nop,sample):
    xt=sample.astype(float)/fps
    yt=sample.astype(float)/fps

    plt.figure(figsize=(13.5,13.5))

    # OLD
    
    
    plt.subplot(711)
    plt.suptitle('allan deviation')
    plt.plot(xt, V[0,:], 'b',linewidth=1.0 )
    plt.ylabel( 'centrex',fontsize=13)


    plt.subplot(712)
    plt.plot(yt,V[1,:],'r',linewidth=1.0 )
    plt.ylabel( 'centrey',fontsize=13)

   
    plt.subplot(713)
    plt.plot(xt,V[2,:],'b',linewidth=1.0 );
    plt.ylabel( 'width',fontsize=13);


    plt.subplot(714)
    plt.plot(yt,V[3,:],'r',linewidth=1.0 )
    plt.ylabel( 'height',fontsize=13)

    
    plt.subplot(715)
    plt.plot(yt,V[4,:],'m',linewidth=1.0 )
    plt.ylabel( 'intensity',fontsize=13)

    
    plt.subplot(716)
    plt.plot(yt,V[5,:],'b',linewidth=1.0 )
    plt.ylabel( 'centrex 2DFFT',fontsize=13)

    
    plt.subplot(717)
    plt.plot(yt,V[6,:],'r',linewidth=1.0 )
    plt.ylabel( 'centrex 2DFFT',fontsize=13)
    plt.xlabel('time(s)')
    
    plt.show()


def Plot_frequencies(freq,nop,fps):

    ind = np.arange(2,int(nop/2)+1,1)
    xf=(np.arange(1,nop+1,1.)/nop)*fps
    f=xf[ind]
    

    plt.figure(figsize=(13.5,13.5))

    # OLD
    

    
    plt.subplot(711)
    plt.suptitle('Frequencies')
    plt.plot(f,freq[0,ind], 'b',linewidth=0.5  )
    plt.ylabel( 'centrex',fontsize=13)
    
    plt.subplot(712)
    plt.plot(f,freq[1,ind],'r',linewidth=0.5)
    plt.ylabel( 'centrey',fontsize=13)
    
    plt.subplot(713)
    plt.plot(f,freq[2,ind],'b',linewidth=0.5 )
    plt.ylabel( 'width',fontsize=13)

    
    plt.subplot(714)
    plt.plot(f,freq[3,ind],'r',linewidth=0.5 )
    plt.ylabel( 'height',fontsize=13)

    
    plt.subplot(715)
    plt.plot(f,freq[4,ind],'m',linewidth=0.5 )
    plt.ylabel( 'intensity',fontsize=13)

    
    plt.subplot(716)
    plt.plot(f,freq[5,ind],'b',linewidth=0.5 )
    plt.ylabel( 'centrex 2DFFT',fontsize=13)
    
   
    plt.subplot(717)
    plt.plot(f,freq[6,ind],'r',linewidth=0.5 )
    plt.ylabel( 'centrey 2DFFT',fontsize=13)
    plt.xlabel('frequency (Hz)')
    plt.show()

#############################################################################
#pr = cProfile.Profile()
#pr.enable()

def ald_square(filename,fps,angle_range,filtersize,N_o_Ad,threspara):
    
    ''' ald_square(foldername,fps,angle_range,twoDfft,filtersize,N_o_Ad,threspara)
    
            This Function is used to compute vibration frequencies and allan
            deviation of the square beams.Use decide_varibles to find the varibles you need before using this.
            
            Input:
            foldername  = the folder of the file.  
            fps = number of frames per second. you can input nan if the fps is
                among 'fps=200,bin=1x1; fps=400,bin=2x2;or fps=713,bin=4X4'. otherwise
                input the real fps value. 
            angle_range= the range of angles fitting(in degree)
                        (suggest:the biggest angle you estimate+5)(the more the better but
                        slower as well~~~)
            twoDfft: if twoDfft=1, the position changes is computed in the method
                    of 2DFFT, otherwise it is computed in the edge detecting method.
            filtersize=the size of FFt band pass filter. (recomand 7)
                        (suggest:the biggest angle+10)
            N_O_Ad=the number of different intergration time you want.(but the generated
                            array will has a slightly difference length, normally 1~4 more than your
                            input number.
            threspara=normal is 1, means set the pixels with intensity lower than half maximum.
                        can set it to 1.1-1.5 if the edge of the image is blur(but sometimes
                        will make a hole inside the image.....)
    
        '''
    
    
    t = time.time()
############################################################################
    
#    ####Variables
#    #please remeber the direction of '/....' (in matlab its'\...'
#    filename="S:/Technical/Diagnostics/Summer Student/Echo/LabCam/frame_155"
    
    
#    angle_range=7
#    filtersize=7
    
    
#    fps=np.nan;
    
    
#    #set the elements that lower than threshold to zero,threshold para normally take 1
#    #but for images with blur edge,a higher threspara (1.0~1.5)will works better.
#    threspara=1.0;
    
#    N_o_Ad=100;
    
    
############################################
    
    
    [data,nop,fps]=fs.hdf5reader(filename,fps)
    
    
################ make it faster here! 
    nop=100;
    data = data[:nop]
##############
    
    
    N_R=2                              #the number of images that used to compute ratoations
    #computing ratations of N_R images 
    angles=fs.average_angles(N_R,data,angle_range,filtersize,threspara)
    
    
    
    sx=3
    sy=3
    #compute the reference of 2Dfft
    im0=data[0,:,:]
    im0=np.array(im0,dtype=float)
    s=im0.shape
    fftim0=F.fftshift(F.fft2(im0))
    phim0=np.angle(fftim0[(int(s[0]/2)-sy):(int(s[0]/2)+sy),(int(s[1]/2-sx)):(int(s[1]/2)+sx)])
    
    
    
    
    
    
    
    #x values of ald
    
    sample=fs.generate_sample_array(N_o_Ad,nop)
    
    
    
    a=np.zeros((7,len(sample)))
    NN=0
    
    
    
    #calculate ald for different number of pics(numbers stored in 'sample')
    for n in sample:
    
        
        a_sum=0
        a_length=0
        #moving the 'window'
        for z in np.arange(0,sample[NN],3):
            STACK_c=np.zeros((np.floor(nop/n),s[0],s[1]))
            NNN=0
            #store the mean values of images in one stack
            for m in np.arange(z,nop-n+1,n):
                STACK_XXX=data[m:m+n,:,:]
                STACK_c[NNN,:,:]=np.mean(STACK_XXX,0)
                NNN=NNN+1
                
            V=np.zeros((7,NNN))
            
            #analyse the images in Stack_c
            for l in range(NNN):             
                im=STACK_c[l,:,:]
                im=np.array(im,dtype=float)
                
                
                #graph here please put 0,otherwise will generate numerous images!!!...
                graph=0;
                ls = fs.analy_square(im,angles,graph,threspara)
                
                #check whether the fitting is inside the image 
                if np.isnan(sum(ls))==True:
    
                    print('Image%04d didn''t work./n'%n)
    
                    print('please move the camera if the image is on the edge. /n')
                    print('Or choose higher thresholding value if only the blur edge of the square is out of picture.')
    
                V[0:5,l]=ls
                V[5:7,l]=fs.fft_2D_squar_beam(im,phim0,sx,sy)
        
                
            #compute the ald for each 'window' then compute the mean of different 'window's
            a_abs = abs(np.diff(V))
            a_sum = a_sum + np.sum(a_abs,1)
            a_length = a_length + np.size(a_abs, 1)
    
        #store ald resultes in a
        a[:,NN] = a_sum / a_length
        NN=NN+1
    
    #compute the params for nop images
    OV=np.zeros((7,nop))
    
    for n in range(nop):
        
        im =data[n,:,:]
        im=np.array(im,dtype=float)
        ls = fs.analy_square(im,angles,0,threspara)
        
        #check whether the fitting is inside the image 
        if np.isnan(sum(ls))==True:
    
            print('Image%04d didn''t work./n'%n)
    
            print('please move the camera if the image is on the edge. /n')
            print('Or choose higher thresholding value if only the blur edge of the square is out of picture.')
    
        OV[0:5,n]=ls
        OV[5:7,n]=fs.fft_2D_squar_beam(im,phim0,sx,sy)
    
    #using FFT to compute the frequencies of the osillation
    
    freq=np.zeros((7,nop))
    
    for n in range(7):                        #store the params
        freq[n,:]=abs(F.fft(OV[n,:]))     
    
    
    
    
    #plot the frequencies of how the beam params changing 
    
    Plot_frequencies(freq,nop,fps)
    
    
    #plot results       
    ald_plots(a,fps,nop,sample)  
    
    print 'computation takes',time.time() - t,'sec'      
############################################################################

#pr.disable()
#s = StringIO.StringIO()


#ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
#ps.print_stats(60)


#print s.getvalue()

################################################################################


ald_square("S:/Technical/Diagnostics/Summer Student/Echo/LabCam/frame_155",np.nan,20,4,20,1.0)
    