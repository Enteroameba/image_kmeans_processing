
#-------------------------------librerias------------------------------------------------------
import cv2 as cv # libreria para el procesamiento de imagenes
import numpy as np # no se que haga pero funciona x
import glob # liberia para leer archivos de las carpetas
import os # libreria para establecer y moverse del directorio de trabajo
import pandas as pd # esta libreria es para generar el data frame del resultado final
import matplotlib.pyplot as plt # esta libreria es para plotear dentro de la consola en python
import matplotlib.image
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw # esta libreria hace el recorte de la parte de enmedio


# directorio de trabajo de donde se obtendran las imagenes a procesar
os.chdir("C:\\Users\\luisd\\Documents\\GitHub\\image_kmeans_processing\\data\\raw")

#------------------------------variables-------------------------------------------------------
n_K = 2 # esta variable guarda el numero de colores que se buscaran en la imagen

df=pd.DataFrame() # este dataframe vacio guarda los resultados originales

#----------------------------------------------------------------------------------------------

# se genera una lista de todas las fotos que se encuentran en la carpeta de input
img_list = glob.glob('*.jpg')

#-------------------------definicion de funciones----------------------------------------------

# funcion para convertir RGB a HEX
def rgb_to_hex(rgb_color:int)->str:
    hex_color = "#"
    for i in rgb_color:
        i = int(i)
        hex_color += ("{:02x}".format(i))
    return hex_color

# funcion para localizar la region de interes (ROI)
def Crop_plate(img:str)->np.ndarray: 
    # reescalado de la imagen a un tamano fijo de 1000x1000
    Input_img=cv.imread(img)
    Input_img=cv.resize(Input_img,(1000,1000), interpolation=cv.INTER_AREA)
    
    # convertir la imagen en un arreglo de 3 dimensiones (ancho x largo x rgb)
    Work_img=Input_img.reshape((Input_img.shape[0])*Input_img.shape[1],3)
    Work_img=np.float32(Work_img)

    # criterios utilizados para el algoritmo de kmeans clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    K = 2 
    attempts=10
    
    # localizacion de los centros de la imagen (pixel mas recurrente)
    _,label,center=cv.kmeans(Work_img,K,None,criteria,attempts,cv.KMEANS_PP_CENTERS)
    center = np.uint8(center)

    # reconstruccion de la imagen utilizando los centros
    res=center[label.flatten()]
    result_image=res.reshape((Input_img.shape))
    
    # se convierte la imagen resultante de RGB a escala de grises
    result_image=cv.cvtColor(result_image,cv.COLOR_RGB2GRAY)

    # se genera una imagen binaria utilizando el algoritmo de otsu
    _,otsu=cv.threshold(result_image,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    # se utiliza la imagen binaria para encontrar los bordes con el algoritmo de canny
    edge=cv.Canny(otsu,10,20)

    # se identifican los maximos y minimos (los bordes de la imagen)
    edge_coordinates = np.argwhere(edge>0)
    y1,x1=edge_coordinates.min(axis=0)
    y2,x2=edge_coordinates.max(axis=0)

    # se recorta la imagen original con los bordes localizados
    ROI=Input_img[y1:y2,x1:x2]

    # se reescala la imagen resultante a un tamano de 500x500 px
    ROI=cv.resize(ROI,(500,500), interpolation=cv.INTER_AREA)

    cv.imwrite(img,ROI)
    # si todo sale bien se genera el siguiente mensaje
    print("ROI Localizada")
    return ROI

# funcion para recortar el centro de la imagen en forma circular
def Center_Crop(img):
    # se carga la imagen a trabajar con la libreria Image
    Input_img = Image.open(img)
    height,width = Input_img.size
    lum_img = Image.new('L',[height,width],0)
    draw = ImageDraw.Draw(lum_img)
    
    # se realiza un corte circular utilizando las coordenadas indicadas 
    draw.pieslice([(10,10), (490,490)], 0, 360, fill=255, outline='black')
    img_arr = np.array(Input_img)
    lum_img_arr = np.array(lum_img)
    final_img_arr = np.dstack((img_arr,lum_img_arr))
    
    # se guarda la imagen resultante utilizando la paqueteria de matplot
    matplotlib.image.imsave(img,final_img_arr)

    # se convierte la imagen a RGB utililzando la libreria Image
    Input_img = Image.open(img)
    Input_img = Input_img.convert('RGB')

    # se toma como referencia el pixel de la esquina para hacer un floodfill 
    reference1 = (20,20)
    color_reference = (0,0,0)
    bucket_img = ImageDraw.floodfill(Input_img, reference1, color_reference, thresh=50)

    # se guarda la imagen resultante
    Output_img = Input_img.save(img)

    # si todo sale bien despliega el siguiente mensaje
    print("Centro Recortado")

# funcion para realizar la segmentacion de la imagen
def Image_segmentation(img):
    # se carga la imagen a trabajar con la libreria de opencv
    Input_img=cv.imread(img)

    # se toma una muestra de la imagen para realizar la segmentacion
    Sample=Input_img[200:300,50:320]

    # se duplica la imagen y se convierten ambas al espacio HSV
    hsv_img=cv.cvtColor(Input_img,cv.COLOR_BGR2HSV)
    hsv2_img=Sample
    hsv2_img=cv.cvtColor(hsv2_img,cv.COLOR_BGR2HSV)

    # se dividen los canales individuales y se aplica el algoritmo de otsu en cada uno
    h,s,v=cv.split(hsv2_img)
    ret_h, _= cv.threshold(h, 0,177, cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret_s, _= cv.threshold(s, 0,255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret_v, _= cv.threshold(v, 0,255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    # otsu regresa un valor de treshold en el cual la imagen se segmenta en dos entidades
    ret_h=int(ret_h)
    ret_s=int(ret_s)
    ret_v=int(ret_v)

    # se utilizan los threshold encontrados para generar una mascara de segmentacion
    lower = np.array([ret_h,0,ret_v])
    upper = np.array([177,ret_v,255])

    # se realiza la segmentacion de la imagen utilizando la mascara generada
    mask=cv.inRange(hsv_img,lower,upper)
    Segmented_img=cv.bitwise_and(Input_img,Input_img,mask=mask)

    # se guarda el resultado final
    cv.imwrite(img,Segmented_img)

    # si todo sale bien imprime el siguiente mensaje 
    print("Imagen segmentada")

# funcion para reconstruir la imagen utilizando los el numero de centros en n_K
def Image_reconstruction(img):
    # se carga la imagen que se va a trabajar 
    ROI=cv.imread(img)

    # se convierte la imagen de BGR a RGB
    ROI=cv.cvtColor(ROI,cv.COLOR_BGR2RGB)

    # se establecen los criterios con los cuales trabajara el algoritmo de kmeans
    attempts=20
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.1)

    # se convierte la imagen a trabajar a un cubo de dimenciones HxWxR,G,B
    reshape_ROI=np.float32(ROI.reshape(ROI.shape[0]*ROI.shape[1],3))

    # se genera la imagen binaria utilizando los criterios preestablecidos
    ret,label,ROIcenter=cv.kmeans(reshape_ROI,n_K,None,criteria,attempts,cv.KMEANS_PP_CENTERS)
    _,counts =np.unique(label, return_counts=True)

    # se reconstruye la imagen original utilizando los centros localizados
    fin = ROIcenter[label.flatten()]
    final_image = fin.reshape((ROI.shape))
    final_image = cv.cvtColor(final_image, cv.COLOR_BGR2RGB)

    # se generan vectores que almacenaran los valores de los pixeles a medida que se reconstruye la imagen
    HEXcolors = [0]
    Frequency =[0]
    R_array = [0]
    G_array = [0]
    B_array = [0]
    z=0
    y=0
    ROIcenter=np.uint8(ROIcenter)

    # se cuenta a que centro corresponde cada uno de los pixeles en la imagen
    for y in range(n_K):
        HEXcolors.append(rgb_to_hex(ROIcenter[y]))
        Frequency.append(counts[y])
        R_array.append(ROIcenter[y,z])
        G_array.append(ROIcenter[y,z+1])
        B_array.append(ROIcenter[y,z+2])
    result_table = pd.DataFrame({'HEXcolors':HEXcolors,'Frequency':Frequency,'R':R_array,'G':G_array,'B':B_array})
    result_table = result_table.iloc[1: , :]
    
    # el resultado es un dataframe que contiene la informacion correspondiente al color de cada centro y el numero de pixeles
    return result_table

#--------------------------------------------------------------------------------------------

for img in img_list:
    ROI=Crop_plate(img)
    Center_Crop(img)
    Image_segmentation(img)
    df=pd.concat([df,Image_reconstruction(img)], ignore_index=True)

print("lote de fotos analizado estos son los resultados:")
print(df)