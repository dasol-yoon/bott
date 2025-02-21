def load(x, filePath, crop=None):
    '''
    returns pre-simulated PACBED image specified by the variable x
    or simulate on the fly
    
    ----variables----
    x: ndarray of shape []
        [ [thickness [angstrom], tilt] ] #todo: verify the format

    filePath = file path where simulated images will be/are saved

    crop: integer that  will crop the diffraction patterns by
        DP[ crop:-crop, crop: -crop]

    todo: add the abtem simulation version
    '''
    thickness=x[0,0]
    tilt = x[0,1]   
    
    tag = '/{:04d}'.format(int(thickness)) #format the thickness value for file name

    #todo: need to test and finalize the folder structure.
    #probably it's better to include tilt in a file name for continuous tilt guesses
    imgPath = glob.glob(fpath+'Tilt1_{}_'.format(int(tilt))
                        +tag[:4]+'*.tif')[0] #1st in list
    try: #load PACBED from the directory
        pacbed = plt.imread(imgPath)
        pacbed = pacbed[crop:-crop,crop:-crop].astype(np.float64) 
        #if the sample is too noisy or not well simulated, crop out area w/ less info
    except: #handle exceptions
        print('Check the file path again. There is no such file: ' +imgPath)

    #pacbed_sqrt = np.sqrt(pacbed) #added 20241017
    #pacbed_norm = pacbed_sqrt/np.mean(pacbed_sqrt) #added 20241017; "standardize" the data...? Isn't it redundant with the normalization?
    pacbed_norm = (pacbed-np.min(pacbed))/(np.max(pacbed)-np.min(pacbed)) #normalize 0-1
    #pacbed_smol = scipy.ndimage.zoom(pacbed_norm.astype(np.float64),20/pacbed_norm.shape[0],order=1) #Nov12: comment out
    #pacbed_norm = (pacbed_smol-np.min(pacbed_smol))/(np.max(pacbed_smol)-np.min(pacbed_smol)) #normalize 0-1 #Nov12: comment out
    

    #unicode for angstrom
    #print('thickness: {} \u212B  |
    # | shape: {} || filename: '.format(x, pacbed_norm.shape)+imgPath[90:])
    
    return torch.Tensor(pacbed_norm)#.reshape(-1) #PB: change to tensor

def getNumPix(y_pred_raw, numPatches): #number of pixels within one patch
    return y_pred_raw.shape[0]//numPatches

def intoPatches(y_pred_raw, numPatches):
    arr = torch.Tensor([]);
    pixPerPatch = getNumPix(y_pred_raw,numPatches) #number of pixels within one patch
    for i in range(numPatches):
        for j in range(numPatches):
            patch = torch.mean(y_pred_raw[...,pixPerPatch*i:pixPerPatch*(i+1),...,pixPerPatch*j:pixPerPatch*(j+1)]).unsqueeze(-1)
            arr = torch.cat((arr,patch*1e1),dim=-1)
    return arr #unpack part 

#todo: change the name of y_raw to be more intuitive
def pixelSSE(y_pred_raw, y_raw):
    err = y_pred_raw-y_raw # #one image y_obs: image I want to measure
    return err.pow(2).sum().unsqueeze(-1) #np.power(err,2).sum().unsqueeze(-1) 

def patchesSSE(pat, y_patC): #patches of y and RSS in 10 variables
    return (pat-y_patC).pow(2).sum().unsqueeze(-1)

def computeY(y_pred_raw):
    '''
    return variable: torch.Tensor of size [1, patchSize**2 +1]
    '''
    pat = intoPatches(y_pred_raw)
    patchTerm = patchesSSE(pat)
    pixelTerm = pixelSSE(y_pred_raw)
    return torch.cat((pat,pixelTerm-patchTerm),dim=-1).unsqueeze(0) #y_pred


def initY(path, testModeX=None):#image in question. 
    '''
    turn the testModeX on to test the pacakge with a set of X values
    by defining them your own. testModeX = x
    example: initY( np.array( [[250,4,0]] ))
    '''
    if testModeX:
        #toodo: need to check the format and shape
        y_raw = load(testModeX)
    else:
        #todo: define a function to format the text image to match the template
    
    y_patC = intoPatches(y_raw) # compute reference values for patches
    y_obsC = computeY(y_raw) # compute reference values
    return y_raw, y_patC, y_obsC



#todo: set lower and upper limit when it runs