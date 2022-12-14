import cv2
import os
import numpy as np
import pandas as pd
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse, diff

def similarity_check(unknown_img):
    l=[]
    for i in os.listdir('symbols'):
        image_path='symbols/'+i
        img1 = cv2.imread(image_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(unknown_img, cv2.COLOR_BGR2GRAY)
        height, width = img1.shape[:2]
        img2=cv2.resize(img2,(width,height))
        error, diff = mse(img1, img2)
        l.append(error)
    return (l.index(min(l))+1)



output=[]
for i in os.listdir('images'):
    image_path='images/'+i
    image=cv2.imread(image_path)
    height, width = image.shape[:2]
    x=130
    y=90
    h=500
    w=width-x-100
    crop_img = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        # invert
    thresh = 255 - thresh

        # apply horizontal morphology close
    kernel = np.ones((5 ,191), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # get external contours
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
        # draw contours
    result = crop_img.copy()
    temp=[]
    symbols=""
    for cntr in contours:
            # get bounding boxes
        pad = 10
        x,y,w,h = cv2.boundingRect(cntr)
        crop_img = result[y-10:y+h+pad,x:x+w+pad]
    
        height, width = crop_img.shape[:2]
        copy=width
        if height>60:
            if( width<220):
                temp_img=crop_img
                symbols+=str((similarity_check(temp_img)))
            while(width>220 and x<copy):
                w=220
                temp_img = result[y-10:y+h+pad,x:x+w+pad]
                x+=220
                height, width = temp_img.shape[:2]
                symbols+=str((similarity_check(temp_img)))
        text = pytesseract.image_to_string(crop_img)
        temp.append(text)
    temp.append(symbols)
    output.append(temp)
    cv2.destroyAllWindows()



device_name=[]
ref=[]
qty=[]
lot=[]
symbols=[]
for d in output:
    a=[d[i].replace('\n',"") for i in range(1,len(d)) if d[i]]
    for i in range(len(a)):
        if i==2:
            temp=a[i].split(" ")
            ref.append("".join(temp[1:]))
        else:
            temp=a[i].split(":")
            if(temp[0]=='Device Name'):
                t=temp[-1].split(":")
                device_name.append(" ".join([i for i in t[-1].split() if len(i)>1]))
            elif temp[0]=='LOT':
                lot.append(temp[-1])
            elif temp[0]=="Qty":
                qty.append(temp[-1])
            elif i==len(a)-1:
                symbols.append(temp[0])
    



df=pd.DataFrame({'Device Name': device_name, 'REF': ref, 'LOT': lot, 'Qty': qty, 'Symbols':symbols})
print(df)
df.to_csv("Output.csv",index=False)












