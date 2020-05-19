#import libraries
import cv2
from pathlib import Path

#declare path for train labels
path=Path("cxr_sr_segmentation_hr/train/label/")
path=path.glob("*.png")
images=[]

count = 1;
for imagepath in path:
        img=cv2.imread(str(imagepath), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256,256))        
        width = int(img.shape[1])
        height = int(img.shape[0])
        dim = (width, height)
        dim1 = (width//8, height//8) #modify for scale factors 2, 3, 4, and 8
        img1 = cv2.resize(img, dim1)
        # resize image
        up = cv2.resize(img1, dim, interpolation = cv2.INTER_CUBIC)
        images.append(up)
        save_dir = 'cxr_sr_segmentation_scale2/train/label/'
        cv2.imwrite(f'{save_dir}/image_{count}.png', up)
        count = count+1;
        
#%% for masks: keep in HR only do resizing and convert to grayscale
#import libraries
import cv2
from pathlib import Path

#declare path
path=Path("cxr_sr_segmentation_hr/result/GT_mask/")
path=path.glob("*.png")
images=[]

count = 1;
for imagepath in path:
        img=cv2.imread(str(imagepath), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256,256))        
        images.append(img)
        save_dir = 'cxr_sr_segmentation_scale8/result/GT_mask/' #modify to store in datasets for different scale factors
        cv2.imwrite(f'{save_dir}/image_{count}.png', img)
        count = count+1;
        
#%%