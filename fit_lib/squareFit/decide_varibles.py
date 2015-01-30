import numpy as np
import matplotlib.pyplot as plt
import functions_for_beam_analyse as fs
import numpy.fft as F
from scipy.ndimage.interpolation import rotate

def fft_filter(Y,filtersize,nop):
    """fft bandpass filter for finding rotation angles
    Filter_Y=fft_filter(Y,filtersize,nop)
    
    input: 
        Y=the data you want to filtered
        filtersize=the size of the filter,smaller size gives smoother result. 
        NOP=number of points
    output:
        Filter_Y=the Filtered data
        """
    FftY=F.fft(Y)
    FftY[filtersize:(nop-filtersize+1)]=0
    Filter_Y=np.abs(F.ifft(FftY))
    

    
    return Filter_Y


def find_angles(im,angle_range,filtersize,hmxin,threspara):
    
    '''This function is used to determine the rotated angles of each side of the square.
     
    params=find_angles(im,angle_range,filtersize,hmxin,threspara)
    
    
    
    input:
        im=image
        angle_range=the range of rotation
        filtersize=the size of FFt band pass filter. (recomand 7)
        hmxin=half maximum of intensity
        threspara: normal is 1, means set the pixels with intensity lower than half maximum.
                    can set it to 1.1-1.5 if the edge of the image is blur(but sometimes
                    will make a hole inside the image.....)
    output:
        'params=[-angle_side1,-angle_side2,-angle_side3,-angle_side4]'
    
    '''
    #set the rotation angles(in degree)
    rotate_step=0.05
    
    rAngle=np.arange(-angle_range,angle_range+rotate_step,rotate_step)
    NoA=angle_range/rotate_step*2
    xshft=1
    yshft=1
    
    #set zero variables
    RSx_Max = np.zeros(rAngle.size)
    RSx_Min = np.zeros(rAngle.size)
    RSy_Max = np.zeros(rAngle.size)
    RSy_Min = np.zeros(rAngle.size)
    
    #t = time.time()

    for n in range(rAngle.size):
        #rotate the picture
        Rim=rotate(im,rAngle[n],reshape=False)
        
    #set the elements that lower than threshold to zero,threshold para normally take 1
    #but for images with blur edge,a higher threspara will works better.

        Rim[Rim<threspara*hmxin]=0
    
        #circshift the picture by certain pixels and then compute the subtract of
        #the obtained picture and the original one. Then the edge of the squre is
        #shown in the obtained picture.    
                    
        Rimx = np.roll(Rim,xshft, axis=1)
        subRimx=Rimx-Rim
        
        newRimy = np.roll(Rim,yshft, axis=0)
        subRimy=newRimy-Rim          
    
        #sum the subtracted picture over x and y respectively
        RSx=subRimx.sum(axis=0)
        RSy=np.transpose(subRimy.sum(axis=1))                  
                                            
        #the angle that gives us highest maxima or lowest minima is 
        #the angle that the side perpendicular to the axis.      
        RSx_Max[n] = np.amax(RSx)
        RSx_Min[n] = np.amin(RSx)
        RSy_Max[n] = np.amax(RSy)
        RSy_Min[n] = np.amin(RSy)    
                                                                                                                                                                
    #Use Fourier bandpass filter to remove the noise
    Filter_RSx_Max=fft_filter(RSx_Max,filtersize,NoA)
    Filter_RSx_Min=fft_filter(RSx_Min,filtersize,NoA)
    Filter_RSy_Max=fft_filter(RSy_Max,filtersize,NoA)
    Filter_RSy_Min=fft_filter(RSy_Min,filtersize,NoA)
    
        
    plt.figure()
    plt.plot(rAngle,abs(RSx_Max))

    plt.plot(rAngle,abs(Filter_RSx_Max))
    plt.show()
    
    plt.figure()
    plt.plot(rAngle,abs(RSx_Min))

    plt.plot(rAngle,abs(Filter_RSx_Min))
    plt.show()
    
    plt.figure()
    plt.plot(rAngle,abs(RSy_Max))

    plt.plot(rAngle,abs(Filter_RSy_Max))
    plt.show()
    
    plt.figure()
    plt.plot(rAngle,abs(RSy_Min))

    plt.plot(rAngle,abs(Filter_RSy_Min))
    plt.show()

    
    #find the angles that correspond tomin&max values
    angle_side1 = rAngle[np.argmax(Filter_RSx_Max)]
    angle_side2 = rAngle[np.argmax(Filter_RSy_Max)]
    angle_side3 = rAngle[np.argmax(Filter_RSx_Min)]
    angle_side4 = rAngle[np.argmax(Filter_RSy_Min)]                                                                                                                                                                                                                                               
    
    
    if angle_side1==0:
        angle_side1=0.0001
    
    if angle_side3==0:
        angle_side3=0.0001
    
    #print time.time() - t
    
    
    params=[-angle_side1,-angle_side2,-angle_side3,-angle_side4]
    

    return params


def decide_varibles(filename,threspara,angle_range,filtersize):
######################################
    
    '''Use this to find the variables you need!!!!
    
    decide_varibles(filename,datatype,angle_range,filtersize,threspara)
    
    output figures:
    the 1~4 is the plot of 'height of peaks in different rotation angles. before and after filtering.... 
    the first image is the image after remove noise.
    
    input:
    filename:filename of the data
    threspara: propotional to the threshold used to remove the background in the image,
            normally use 1, which means set the pixels with value lower
            than half maximum to 0.
            can set a higher value if the edge is blur.
    angle_range: the angle range of rotating images
                increase the angle_range if the highest peak is close to
                the edge of the plot. otherwise decrese the angle range to
                speed up the fitting
                suggest begain with 10.
    
    filtersize:FFT bandpass filter to remove the noise on the plot of 'height
            of peaks in different rotation angles. A narrower bandpass 
            correspond to a smoothier result, choose the value that you think
            can give you the right peak on the plot.
            suggest begain with 7.'''

#################################################    
    #variables need to be decide
    
#    angle_range=20
#    filtersize=4
#   threspara=1.0
    
    ######################################
    #other variables

    #filename="S:/Technical/Diagnostics/Summer Student/Echo/LabCam/frame_155"

    ###################################

    [data,nop,fps]=fs.hdf5reader(filename,np.nan)    
    im=data[0,:,:]

        
    
    im=np.array(im,dtype=float)
    
    thres_im=im
    
    apxparams=fs.get_apx_params(im,0)
    
    
    hmxin=apxparams[0]
    
    
    #computing ratations of N_R images 
    angles=find_angles(im,angle_range,filtersize,hmxin,threspara)
    
    print 'the angles of the four side are',angles,'respectively'
     
    
    threshold=threspara*hmxin    
    thres_im[thres_im<threshold]=0  
    plt.figure()      
    plt.imshow(thres_im)
    plt.show()
######################################


decide_varibles("S:/Technical/Diagnostics/Summer Student/Echo/LabCam/frame_155",1.0,20,5)