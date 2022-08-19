from tkinter import filedialog
import cv2
import numpy as np
import math
from tkinter import *
from PIL import ImageTk,Image

#Defining the Name and Window Size of the Initial Tkinter Dialog Box
root = Tk()
root.title("Image Editor")
root.geometry('1300x650')

#Function Definitions of all the function Used in the Image Editor
#The load function is defined below:

def open():
    #Defining the set of global variables that would be used in all the functions

    global my_label
    global my_image
    global current_value
    global u #Undo variable
    global ul #Undo_All variable

    #filedialog box library is used to select the type of files(jpg, png etc) with location of the initial directory

    root.filename = filedialog.askopenfilename(initialdir = "", title = "Select a File", filetypes = (("jpg files","*.jpg"),("png files","*.png")))
    img = Image.open(root.filename)

    img = img.resize((800, 600), Image.ANTIALIAS) #All the input images are resized to 800*600 resolution so that the uniformity in presentation is maintained
    my_image = ImageTk.PhotoImage(img) #PIL library is used to display image in the tkinter GUI

    numpy_img = np.array(img) #Image is converted in numpy array for performing operations and enhancements

    current_value = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
    u = current_value # Assigning the image array value to undo all variable
    ul = current_value # Assigning the image array values to undo last variable

    my_label = Label(image = my_image) #Creating the my_label widget and pushing the image for display
    my_label.grid(row=1, column=0, columnspan=9) #Assigning the orientation of the image for display


def log_transform(img):
    global my_label
    global my_image
    global current_value
    global u
    global ul

    u = img #Assigning the current value before transformation to undo variable
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #Converting the BGR image passed to HSV for performing modifications
    img_val = img_HSV[:, :, 2] #Modifying only the value channel of the HSV matrix

    img_val = img_val.astype('uint64') #In order to avoid overflow of due to 8 bit integer first, the values are converted to 64 bit integers
    c = int(255/(math.log(1+np.max(img_val)))) #Log Transformation is Applied and the value of normalization Constant is calculated

    img_val = np.array(c * (np.log(img_val + 1)),dtype='uint8')
    img_HSV[:, :, 2] = img_val #Modified value is then resassigned to the original image


    color_coverted = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2BGR) #HSV image is converted to BGR
    current_value = color_coverted #The global variable val is updated with the new value of transformed image
    color_coverted = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2RGB) #HSV image is converted to RGB

    #RGB image is again converted into PIL form for display into the Tkinter GUI
    disp(color_coverted)

def gamma_click(img):

    #Defining the gamma value window for gamma correction
    frame = Tk()
    frame.title("Gamma Value")

    e = Entry(frame, width=35, borderwidth=5) #Dimensions of the window is defined
    e.pack()

    def value():

        y = e.get()
        gamma_corr(img, float(y)) #Value of input gamma is passed to the Gamma Correction Function

    my_button = Button(frame, text = "Enter Gamma Value" , command = value)
    my_button.pack()


def blur_click(img):

    #Defining the blurring extent of the image using an input window
    frame3 = Tk()
    frame3.title("Blur Value")

    e = Entry(frame3, width=35, borderwidth=5)
    e.pack()

    def value():
        y = e.get()
        blur(img, int(y))#Value of input is passed to the Gamma Correction Function


    my_button = Button(frame3, text="Enter Odd Integer Value", command=value)
    my_button.pack()

def sharp_click(img):

    #Defining the sharpening extent of the image using an input window
    frame2 = Tk()
    frame2.title("Sharpen Value")

    e = Entry(frame2, width=35, borderwidth=5)
    e.pack()

    def value():
        y = e.get()
        sharpen(img, float(y))#Value of input is passed to the Gamma Correction Function


    my_button = Button(frame2, text="Enter Sharpness Amount", command=value)
    my_button.pack()


def gamma_corr(img, gamma=0.5):

    global my_label
    global my_image
    global current_value
    global u
    global ul

    u = img #Current Image value to undo variable
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_val = img_HSV[:, :, 2] #Assigning the value channel to img_val for modifications

    trans_img_value = np.array(255 * (img_val / 255) ** (1/gamma), dtype='uint8') #Performing the gamma Correction Operation
    img_HSV[:, :, 2] = trans_img_value
    color_coverted = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2BGR)

    current_value = color_coverted #Modified image value to Val
    color_coverted = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2RGB)
    disp(color_coverted)

def undo_all():
    global my_label
    global my_image
    global current_value
    global u
    global ul

    current_value = ul #Assigning the initial value of image stored to val variable and displaying
    color_coverted = cv2.cvtColor(current_value, cv2.COLOR_BGR2RGB)
    disp(color_coverted)

def undo():
    global my_label
    global my_image
    global current_value
    global u
    global ul

    current_value = u #Assigning the last value of image stored in u variable and displaying the image
    color_coverted = cv2.cvtColor(current_value, cv2.COLOR_BGR2RGB)
    disp(color_coverted)

    return

def save():

    global my_label
    global my_image
    global current_value
    global u
    global ul

    #Saving the file in the current directory with the following "Saved_Image" name
    file_name = "Saved_Image.jpg"
    cv2.imwrite(file_name,current_value)
    return

def disp(ig):
    global my_label
    global my_image
    global current_value
    global u
    global ul

    #Function used to convert the RGB form of image to PIL for display
    pil_image = Image.fromarray(ig)
    my_image = ImageTk.PhotoImage(pil_image)
    my_label.grid_forget()
    my_label = Label(image=my_image)
    my_label.grid(row=1, column=0, columnspan=9)

def convolve2D(image, kernel,padding):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]
    strides = 1
    # Shape of Output Convolution
    xOutput = int(((xImgShape + 2 * padding) / strides) )
    yOutput = int(((yImgShape + 2 * padding) / strides) )
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image

    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1]:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0]:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x+padding, y+padding] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def blur(image,n):

    global my_label
    global my_image
    global current_value
    global u
    global ul

    u = image
    kernel = (1/n**2) * np.ones((n,n)) #Creating the average pooling kernel with input as kernel size
    padding = int((n-1)/2) #Equal padding is added to all the four sides
    img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    a = img_HSV[:, :, 2]
    output = convolve2D(a,kernel,padding) #Value channel, kernel and padding is sent as an input to the convolve2D function
    out_dim = np.shape(output)

    new = np.zeros((out_dim[0]-2*padding,out_dim[1]-2*padding,3)) #Getting back the number of rows and columns of the output matrix obtained from convolution
    new[:, :, 2] = output[int(padding):int(-1 * padding ), int(padding):int(-1 * padding)]
    new[:,:, 0:2] = img_HSV[:,:,0:2] #Reassigning the value of first 2 vector channels to the new output

    img_HSV = new.astype('uint8') #Reconverting to uint8 data type

    color_coverted = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2BGR)

    current_value = color_coverted
    color_coverted = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2RGB)
    disp(color_coverted)

def sharpen(image,n):

    global my_label
    global my_image
    global current_value
    global u
    global ul

    u = image
    img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]) #Kernel is already defined here
    padding = 1
    a = img_HSV[:, :, 2]
    output = convolve2D(a, kernel,1)
    low = output.min()
    high = output.max()
    out_dim = np.shape(output)
    temp = np.ones((out_dim[0],out_dim[1]))
    temp = temp * low
    output = 255*((output - temp)/high)
    new = np.zeros((out_dim[0] - 2 * padding, out_dim[1] - 2 * padding, 3))
    new[:, :, 2] = a -n* output[int(padding):int(-1 * padding), int(padding):int(-1 * padding)]
    new[:, :, 0:2] = img_HSV[:, :, 0:2]
    img_HSV = new.astype('uint8')

    color_coverted = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2BGR)
    current_value = color_coverted
    color_coverted = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2RGB)
    disp(color_coverted)

def get_histogram(image, bins):
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)

    # loop through pixels and sum up counts of pixels
    for pixel in image:
        histogram[pixel] += 1

    # return our final result
    return histogram


def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)


def hist_EQ(image):

    global my_label
    global my_image
    global current_value
    global u
    global ul

    u = image
    img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    img_val = img_HSV[:,:,2]
    img_val = np.asarray(img_val)
    # put pixels in a 1D array by flattening out img array
    flat = img_val.flatten()


    hist = get_histogram(flat, 256)

    cs = cumsum(hist)

    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()

    # re-normalize the cdf
    cs = nj / N
    cs = cs.astype('uint8')
    img_new = cs[flat]
    img_new = np.reshape(img_new, img_val.shape)

    img_HSV[:,:,2] = img_new
        #Assigning the values of the transformed channel to the original HSV image

    color_coverted = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2BGR)
    # cv2.imshow('a2',color_coverted)
    current_value = color_coverted
    color_coverted = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2RGB)
    disp(color_coverted)

def reduce(image):
    global my_label
    global my_image
    global current_value
    global u
    global ul

    u = image
    img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_val = img_HSV[:, :, 2]
    img_val = (img_val//32) * 32
    img_HSV[:, :, 2] = img_val

    color_coverted = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2BGR)
    current_value = color_coverted
    color_coverted = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2RGB)
    disp(color_coverted)

my_btn_load = Button(root, text = "Load Image", padx = 20, pady = 10,command = open).grid(row = 0 , column = 0)
my_btn_Histogram = Button(root, text = "Equalize Histogram",padx = 17, pady = 10, command = lambda:hist_EQ(current_value)).grid(row = 0 , column = 1)
my_btn_Gamma = Button(root, text = "Gamma Transform",padx = 20, pady = 10, command = lambda:gamma_click(current_value)).grid(row = 0 , column = 2)
my_btn_Log = Button(root, text = "Log Transform",padx = 20, pady = 10, command = lambda:log_transform(current_value)).grid(row = 0 , column = 3)
my_btn_Blur = Button(root, text = "Blur",padx = 45, pady = 10, command = lambda:blur_click(current_value)).grid(row = 0 , column = 4)
my_btn_Sharpen = Button(root, text = "Sharpen",padx = 40, pady = 10, command = lambda:sharp_click(current_value)).grid(row = 0 , column = 5)
my_btn_Reduction = Button(root, text = "Color Reduction",padx = 40, pady = 10, command = lambda:reduce(current_value)).grid(row = 0 , column = 6)
my_btn_Undo = Button(root, text = "Undo",padx = 40, pady = 10,command = undo).grid(row = 0 , column = 7)
my_btn_UndoALl = Button(root, text = "Undo All",padx = 30, pady = 10,command = undo_all).grid(row = 0 , column = 8)
my_btn_Save = Button(root, text = "Save",padx = 40, pady = 10, command = save).grid(row = 0 , column = 9)

root.mainloop()