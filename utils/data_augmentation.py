from scipy import ndimage, misc
import cv2
import numpy as np
from PIL import Image, ImageDraw

nd = 32
an = 15
t_perc = 0.2

# aplicando translação
def translate(nimg):
  img = np.asarray(nimg)
  
  img_width=np.shape(img)[1]
  img_height=np.shape(img)[0]
  
  tx = np.random.randint(int(-img_width*t_perc),int(img_width*t_perc))
  ty = np.random.randint(int(-img_height*t_perc),int(img_height*t_perc))

  translation_matrix = np.float32([ [1,0,tx], [0,1,ty] ])
  img = cv2.warpAffine(img, translation_matrix, (img_width, img_height))
  out_img = Image.fromarray(img)
  return out_img

# aplicando rotações
def rotate(nimg):
  
  img = np.asarray(nimg)
  angle = np.random.uniform(-an, an)
  
  img_width=np.shape(img)[1]
  img_height=np.shape(img)[0]
  
  (cX, cY) = (img_width//2, img_height//2)
  
  M = cv2.getRotationMatrix2D((cX,cY), angle, 1.0)
  cos = np.abs(M[0, 0])
  sin = np.abs(M[0, 1])

  nW = int((img_height * sin) + (img_width * cos))
  nH = int((img_height * cos) + (img_width * sin))

  M[0, 2] += (nW / 2) - cX
  M[1, 2] += (nH / 2) - cY

  img = cv2.warpAffine(img, M, (nW, nH))
  
  out_img = Image.fromarray(img)
  out_img = out_img.resize((img_width,img_height))
  
  return out_img

# horizontal flip na imagem e na bbox correspondente
def h_flip(img):
  img = np.asarray(img)
  img =  cv2.flip(img, 1)
  out_img = Image.fromarray(img)
  return out_img


# variando a iluminação usando hsv da imagem
def hue_transform(img):
  
  img = np.asarray(img)
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  ratio = .5 + np.random.uniform()
  hsv[:,:,2] =  np.clip(hsv[:,:,2].astype(np.int32) * ratio, 0, 255).astype(np.uint8)
  img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
  
  img = Image.fromarray(img)
  
  return img


def cutout(img, bb_shape= (15,15)):
  
  img = np.asarray(img)

  h,w,_ = img.shape
  img = img.copy()
  
  bh, bw = bb_shape
  for n in np.arange(3):
    
    if np.random.uniform() > 0.5:
      # sample a centroid
      cx = np.random.randint(bw,w-bw)
      cy = np.random.randint(bh,h-bh)

      img[cy-bh:cy+bh,cx-bw:cx+bw,:] = np.random.randint(256,size=(3,))

  
  img = Image.fromarray(img)
  
  return img


def zooming(img, bb_shape= (15,5)):
  
  img = np.asarray(img)

  h,w,_ = img.shape
  img = img.copy()
  
  bh, bw = bb_shape
  
    
  if np.random.uniform() > 0.5:
    # sample a centroid
    cx1 = np.random.randint(0,bw)
    cy1 = np.random.randint(0,bh)
    cx2 = np.random.randint(0,bw)
    cy2 = np.random.randint(0,bh)

    img = img[cy1:h-cy2,cx1:w-cx2]

  
  img = Image.fromarray(img)
  img = img.resize((w,h), Image.ANTIALIAS)
  return img

def augment(nimg,methods):

  #cutout
  if 'cutout' in methods:
    if np.random.uniform() > 0.5 :
      nimg  = cutout(nimg)
      # print('cutout: ', type(nimg))
  # brightness  
  if 'brightness' in methods:
    if np.random.uniform() > 0.5 :
      nimg = hue_transform(nimg)
      # print('brightness: ', type(nimg))

  # horizontal flip  
  if 'horizontal_flip' in methods:
    if np.random.uniform() > 0.5:
      nimg = h_flip(nimg)    
      # print('horizontal_flip: ', type(nimg))
  
  # zooming  
  if 'zooming' in methods:
    if np.random.uniform() > 0.5:
      nimg = zooming(nimg)    
      # print('zooming: ', type(nimg))


  # zooming  
  if 'rotate' in methods:
    if np.random.uniform() > 0.5:
      nimg = rotate(nimg) 
      # print('rotate: ', type(nimg))


  # zooming  
  if 'translate' in methods:
    if np.random.uniform() > 0.5:
      nimg = translate(nimg)  
      # print('translate: ', type(nimg))


  return nimg