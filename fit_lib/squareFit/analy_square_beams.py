import numpy as np
import matplotlib.pyplot as plt
import time
import numpy.fft as F
import functions_for_beam_analyse as fs
import pdb

#import cProfile, pstats, StringIO
#import re


    
def time_plots(V,fps,nop):
    xt=np.arange(1,nop+1,1.)/fps
    yt=np.arange(1,nop+1,1.)/fps
    
    plt.figure(figsize=(13.5,13.5))

    # OLD

    
    plt.subplot(711)
    plt.suptitle('Time plots')
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



def analy_square_beams(filename,fps,angle_range,threspara,showfitting,filtersize):

    '''analy_square_beams(filename,fps,angle_range,filtersize,threspara)
    
    This function is used to analyse images square beams.
        
        This function is based on edge detecting method and 2D FFT of image. 
        
        edge detcting:
            find the slope and positions of the squares etc...
                
                        vertex3         vertex 2
                       \   side 2   /     
                        \_________ /
                        /         |
                side 3 /          |  side 1
                      /           |  
                     /____________|
                     |     side 4  \
                 vertex 4        vertex 1    
                    
        
        
        2D FFT:
        According to the definition of 2D FT, the change of the 
        position of the image result in a linear change of phase. By ignoring the
        background noise, the vabrition of the beam can be regarded as the 'vabrition'
        of the whole image. By computing the average phase shift of the 2D FFT of
        images, we get the changes of the central position of beams.
        
        
        Since the shape of the square does not change a lot during the period, such we only compute the 
        average angles of 10 pictures.
        
        
        There are two output images in this function.
        The first output image shows how the intensity, position, size change with time.  
        "centrex","centrey","width" and "height" are obtianed by finding the edge of
        square beams.The "intensity" is the overall intensity of the image. While
        the "centrex_fft"&"centrey_fft" is obtained by 2D fft transformation,which
        focus on the overall centre of the beam.
        
        The second output image is the 1D fft of the first one, which indicates
        the frequency of vibrations.
        
        filename  = the name and path of the HDF5 file.  
        fps = number of frames per second. you can input nan if the fps is
            among 'fps=200,bin=1x1; fps=400,bin=2x2;or fps=713,bin=4X4'. otherwise
            input the real fps value. 
        angle_range= the range of angles fitting(in degree)
                    (suggest:the biggest angle you estimate+10)(the more the better but
                    slower as well~~~)
        threspara:normally is 1, means set the pixels with intensity lower than half maximum.
                        can set it to 1.1-1.5 if the edge of the image is blur(but sometimes
                        will make a hole inside the image.....)
        showfitting: if showfitting =1, then the fitting of the suquare is
                    displayed when the code is running.
        filtersize=the size of FFt band pass filter. (recomand 7)   
            
        Output:
        a figure of position,size and intensity change, a figure of corresponding
        frequency


    '''    
    ###################################################################
    
    
    #pr = cProfile.Profile()
    #pr.enable()
    t = time.time()
    ###################################################################
    
    
#    #variables
#    filename="S:/Technical/Diagnostics/Summer Student/Echo/LabCam/frame_155"
#    
#    angle_range=10
#    filtersize=4
#   showfitting=0;
    
#    #set the elements that lower than threshold to zero,threshold para normally take 1
#    #but for images with blur edge,a higher threspara (1.0~1.5)will works better.
#    threspara=1.0;
    
    
#    #   fps = number of frames per second. you can input 'np.nan' if the fps is
#    #      among 'fps=200,bin=1x1; fps=400,bin=2x2;or fps=713,bin=4X4'. otherwise
#    #       input the real fps value. 
#    fps=np.nan
#    
#    #####################################################################
    [data,nop,fps]=fs.hdf5reader(filename,fps)
    
    ############
    #use less figure to test the fitting
    #nop=200
    #data=data[:nop]
    
    #########################
    #the number of pictures for angle computation,normally put 10, but can be put as 1
    #to make the code faster when teast the code.
    N_R=10
    
    #size of the phase image been computed
    sx=3;
    sy=3;
    
    
    #computing ratations of N_R images 
    angles=fs.average_angles(N_R,data,angle_range,filtersize,threspara)
    
    
    
    #compute the reference of 2Dfft
    im0=data[0,:,:]
    im0=np.array(im0,dtype=float)
    
    s=im0.shape
    fftim0=F.fftshift(F.fft2(im0))
    phim0=np.angle(fftim0[(int(s[0]/2)-sy):(int(s[0]/2)+sy),(int(s[1]/2-sx)):(int(s[1]/2)+sx)])
    
    
    #compute the params for nop images
    V=np.zeros((7,nop))
    
    #the images want to plot, put the number of the images, eg,[1,2,3] 
    if showfitting==1:
        showfitting=[0,1,2]
    else :
        showfitting=[]
        
    for n in range(nop):
        
        im =data[n,:,:]
        im=np.array(im,dtype=float)
        
        
        if n in showfitting:
            graph=1
        else:
            graph=0
    
        ls = fs.analy_square(im,angles,graph,threspara)
        
        #check whether the fitting is inside the image 
        if np.isnan(sum(ls))==True:
    
            print('Image%04d didn''t work./n'%n)
    
            print('please move the camera if the image is on the edge. /n')
            print('Or choose higher thresholding value if only the blur edge of the square is out of picture.')
    
        V[0:5,n]=ls
        V[5:7,n]=fs.fft_2D_squar_beam(im,phim0,sx,sy)
        
    #plot the params of nop images vs time
    
    time_plots(V,fps,nop)
    
    #using FFT to compute the frequencies of the osillation
    
    freq=np.zeros((7,nop))
    
    for n in range(7):                        #store the params
        freq[n,:]=abs(F.fft(V[n,:]))    
    
    
    #plot the frequencies of how the beam params changing 
    
    Plot_frequencies(freq,nop,fps)
    
    print time.time() - t
    
    ###########################################################
    
    #pr.disable()
    #s = StringIO.StringIO()
    
    
    #ps = pstats.Stats(pr, stream=s).sort_stats('time')
    #ps.print_stats(60)
    
    #print s.getvalue()
    ########################################

analy_square_beams("S:\Technical\Diagnostics\Summer Student\2014 Echo\Summer placement\LabCam\frame_155",np.nan,20,1.1,1,4)