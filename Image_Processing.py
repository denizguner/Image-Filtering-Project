#!/usr/bin/env python3

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import math
from PIL import Image
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox

class invalidBoundryBehaviour(Exception):
    pass

def get_pixel(image, x, y):
    image_width = image['width']
    return image['pixels'][y*image_width+x] 
        
def set_pixel(image, x, y, c):
    image_width = image['width']
    image['pixels'][y*image_width+x] = c 


def apply_per_pixel(image, func):
    result = {
        'height': image['height'],
        'width': image['width'], 
        'pixels': image['pixels'][:] 
    }
    for x in range(image['width']):
        for y in range(image['height']): 
            color = get_pixel(image, x, y)
            newcolor = func(color)
            set_pixel(result, x, y, newcolor) 
    return result


def inverted(image):
    return apply_per_pixel(image, lambda c: 255-c)

def outOfBounds(width, height, x, y):
    """
    Given coordinates x,y returns a bool, which is True if the pixel is out of bounds 
    for the given width and height, and False Otherwise
    
    type(width, height, x, y) = nums
    type(result) = Boolean
    """
    if x < 0 or y < 0 or x >= width or y >= height:
        return True
    return False

def coordinatesToCheck(kernelSideLength, x , y):
    """
    Given coordinates x and y, and a kernelSideLength, returns a list of x,y
    coordinates that need to be checked for the kernel calculation
    """
    
    coordinates = []
    kernel_coordinates = []
    for x_candidate in range(-1 * kernelSideLength//2 + 1, kernelSideLength//2 + 1):
        for y_candidate in range(-1 *kernelSideLength//2 + 1, kernelSideLength//2 + 1):
            coordinates.append((x + x_candidate, y + y_candidate))
            
    return (coordinates)
            

def outOfBoundsValue(image, coordinate, boundary_behavior):
    """
    Given the pixels of an image, and out of bounds coordinates, returns a value to use for
    kernel calculations based on mode
    """

    if boundary_behavior == 'zero':
        return 0
    
    
    if boundary_behavior == 'extend': 
        x = coordinate[0]
        y = coordinate[1]
        
        #One mode of failure here is checking if a value is less then the upper bound
        #but not checking if it is above the lower bound
        
        #Can be fixed with correct ordering or implicit checks
        
        if x <= 0 and y <= 0: #top left corner
            return get_pixel(image, 0, 0)
        if x <= 0 and y < image['height']: #left side
            return get_pixel(image, 0, y)
        if x <= 0 and y >= image['height']: #bottom left corner
            return get_pixel(image, 0, image['height'] - 1)
        if x < image['width'] and y <= 0: #top side
            return get_pixel(image, x, 0)
        if x >= image['width'] and y <= 0: #top right corner
            return get_pixel(image, image['width']-1, 0)
        if x < image['width'] and y >= image['height']: #bottom side
            return get_pixel(image, x , image['height']-1)
        if x >= image['width'] and y >= image['height'] - 1: #bottom right corner
            return get_pixel(image, image['width']-1, image['height']-1)
        if x >= image['width'] and y < image['height']: #left side
            return get_pixel(image, image['width']-1, y)
        
    if boundary_behavior == 'wrap': #The modulo operation can be used to calculate where the pixel should end up
        wrap_x = coordinate[0]%image['width']
        wrap_y = coordinate[1]%image['height']
        return get_pixel(image, wrap_x, wrap_y)
    
    #Throw an error for other boundry behaviours
    raise invalidBoundryBehaviour 
    
    
    
    
def correlate(image, kernel, boundary_behavior, mode = 'notEdge'):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings 'zero', 'extend', or 'wrap',
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of 'zero', 'extend', or 'wrap', return
    None.

    This process does not mutate the input image; rather, it creates a
    separate structure to represent the output.

    Kernel will be a list, formatted similarly to image['pixels'].
    """
    if boundary_behavior not in {'zero', 'extend', 'wrap'}:
        return None
    
    newImage = {'width': image['width'], 'height': image['height'], 'pixels': image['pixels'][:]}
    kernelSideLength = round(len(kernel)**(1/2))
    
    for x in range(image['width']): #Iterate trough every pixel
        for y in range(image['height']):
            pixelsToCheck = coordinatesToCheck(kernelSideLength, x, y) #Get pixels involved in calculation
            totalPixelValue = 0 #New value of pixel, to be calculated
            kernelCounter = 0
            for coordinate in pixelsToCheck:
                value = 0
                if outOfBounds(image['width'], image['height'], *coordinate):
                    value = outOfBoundsValue(image, coordinate, boundary_behavior)
                else:
                    value = get_pixel(image, *coordinate)

                totalPixelValue += (value * kernel[kernelCounter])
                kernelCounter += 1
            newImage['pixels'][y*image['width']+x] = totalPixelValue
    
    if mode == 'edge':
        return newImage
                     
    round_and_clip_image(newImage)
    return newImage
    


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the 'pixels' list are all
    integers in the range [0, 255].
    """
    pixels = image['pixels']
    for i in range(len(pixels)):
        if pixels[i] > 255:
            image['pixels'][i] = 255
        elif pixels[i] < 0 :
            image['pixels'][i] = 0
        else:
            image['pixels'][i] = round(pixels[i])


# FILTERS

def kernelGenerator(n, mode = 'blur'):
    """
    Generates an n by n kernel for use in the blur mask.
    """
    result = []
    value = 1/(n**2)
    for i in range(n**2):
        result.append(value)
    if mode == 'blur':
        return result
    elif mode == 'sharpen':
        for i in range(len(result)):
            result[i] *= -1
        result[(n**2)//2] += 2
        return result

def blurred(image, n):
    """
    Return a new image representing the result of applying a box blur (with
    kernel size n) to the given input image.

    This process does not mutate the input image; rather, it creates a
    separate structure to represent the output.
    """
    kernel = kernelGenerator(n)
    newImage = correlate(image, kernel, 'extend')
    return newImage


def sharpened(image, n):
    """
    Return a new image representing the result of applying a box sharpen (with
    kernel size n) to the given input image.

    This process does not mutate the input image; rather, it creates a
    separate structure to represent the output.
    """
    kernel = kernelGenerator(n, 'sharpen')
    newImage = correlate(image, kernel, 'extend')
    return newImage

def edges(image):
    """
    Return a new image representing the result of applying a edge detection mask to the given input image.

    This process does not mutate the input image; rather, it creates a
    separate structure to represent the output.
    """
    x_kernel = [-1,0,1,-2,0,2,-1,0,1]
    y_kernel = [-1,-2,-1,0,0,0,1,2,1]
    
    x_image = correlate(image,x_kernel,'extend','edge')
    y_image = correlate(image,y_kernel,'extend','edge')
    
    final_image = {'height':image['height'], 'width':image['width'], 'pixels': x_image['pixels'][:]}
    
    for i in range(len(final_image['pixels'])):
        final_image['pixels'][i] = round(((x_image['pixels'][i]**2)+(y_image['pixels'][i]**2))**(1/2))
        
    round_and_clip_image(final_image)
    
    return final_image
    
def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """

    def color_filt(image):
        
        greyscaleImages = colorSplitter(image)
        
        for i in range(3):
            greyscaleImages[i] = filt(greyscaleImages[i])
            
        filteredImage = {'width':image['width'], 'height':image['height'], 'pixels':[(greyscaleImages[0]['pixels'][i]
                                                                                      , greyscaleImages[1]['pixels'][i]
                                                                                      , greyscaleImages[2]['pixels'][i])                                                                  for i in range(len(image['pixels']))]}
        return filteredImage

    return color_filt

def make_blur_filter(n):
    
    def newBlurred(image):
        return blurred(image,n)    
    
    return newBlurred



def make_sharpen_filter(n):
    
    def newSharpened(image):
        return sharpened(image,n)    
    
    return newSharpened


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    def cascadedFilters(image):
        for filt in filters:
            image = filt(image)
        return image
    
    return cascadedFilters


# SEAM CARVING

# Main Seam Carving Implementation

def seam_maker(ncols):
    "Returns a function object of the seam_carving function"
    
    def seam(image):
        image = seam_carving(image, ncols)
        return image
    
    return seam

def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image.
    """
    ans = {'height':image['height'],'width':image['width'],'pixels':image['pixels'][:]}
    def one_carve(image):
        grey = greyscale_image_from_color_image(image)
        energy = compute_energy(grey)
        cem = cumulative_energy_map(energy)
        seam = minimum_energy_seam(cem)
        final = image_without_seam(image, seam)
        final['width'] = final['width'] - 1
        return final
    
    for i in range(ncols):
        ans = one_carve(ans)
    return ans


#Helper Functions for Seam Carving


def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    return {'width':image['width'], 'height':image['height'], 'pixels':[round(0.299*color[0]+0.587*color[1]+0.11*color[2]) for color in image['pixels']]}
    

def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function.

    Returns a greyscale image (represented as a dictionary).
    """
    return edges(grey)

def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map".

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    for i in range(1,energy['height']):
        for j in range(energy['width']):
            energy['pixels'][i*energy['width']+j] = energy['pixels'][i*energy['width']+j] + min(energy['pixels'][(i-1)*energy['width']+j],energy['pixels'][(i-1)*energy['width']+j+1],energy['pixels'][(i-1)*energy['width']+j-1])       
    return energy

def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam.
    """
    bottom_row = cem['pixels'][(cem['width']*(cem['height']-1)):]
    starting_x = 0
    lowest = math.inf
    for x in range(cem['width']):
        if bottom_row[x] < lowest:
            starting_x = x
            lowest = bottom_row[x]
    ans = [starting_x]
    
    for i in range(cem['height']-2,-1,-1):
        
        tofind = ans[-1]
        left = math.inf
        right = math.inf
        if tofind != 0:
            left = cem['pixels'][i*cem['width']+(tofind-1)]
        if tofind != cem['width'] - 1:
            right = cem['pixels'][i*cem['width']+(tofind+1)]
        mid = cem['pixels'][i*cem['width']+(tofind)]
        
        toappend = tofind-1
        if mid < left:
            toappend = tofind
        if right<mid:
            if right<left:
                toappend = tofind + 1
            else:
                toappend = tofind - 1
                
        ans.append(toappend)
    ans.reverse()
    return ans 

def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    ans = {'width':image['width'], 'height':image['height'], 'pixels':[]}
    for row in range(image['height']):
        current_row = image['pixels'][row*image['width']:(row+1)*image['width']]
        current_row.pop(seam[row])
        ans['pixels'].extend(current_row)
    return ans


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES


def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img = img.convert("RGB")  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        w, h = img.size
        return {"height": h, "width": w, "pixels": pixels}


def save_color_image(image, filename, mode="PNG"):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode="RGB", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError('Unsupported image mode: %r' % img.mode)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_greyscale_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()

def colorSplitter(image):

    """
    Given an rgb image, split up the red, green and blue values into seperate greyscale images.
    """
    red_greyscale = {'width':image['width'], 'height':image['height'], 'pixels':[]  
        }
    green_greyscale = {'width':image['width'], 'height':image['height'], 'pixels':[] 
        }
    blue_greyscale = {'width':image['width'], 'height':image['height'], 'pixels':[] 
        }
    for pixel in image['pixels']:
            red_greyscale['pixels'].append(pixel[0])
            green_greyscale['pixels'].append(pixel[1])
            blue_greyscale['pixels'].append(pixel[2]) 
        
    return [red_greyscale,green_greyscale,blue_greyscale]

if __name__ == "__main__":
    
    ROOT = tk.Tk()

    ROOT.withdraw()
    # the input dialog
    USER_INP = simpledialog.askstring(title="Image Location",
                                  prompt="Please input location of image to be edited, including the file extension:")
    
    
    USER_INP2 = simpledialog.askstring(title="Color Mode",
                                  prompt="Please input if you want your image to be treated as RGB or BW:")
    
    
    if USER_INP2 == 'RGB' or USER_INP2 == 'rgb':
        USER_INP3 = simpledialog.askstring(title="Filters",
                                      prompt="Please choose filter to apply to your image, choose from invert,blur,sharpen,detect_edges,and seam_carving, for a cascaded filter effect, please input multiple filters with comma and no space between:")
        
        Im = load_color_image(USER_INP)
        
        Filters = USER_INP3.split(",")
        
        ToCascade = []
        
        
        for FILTER in Filters:
            if FILTER == 'blur':
                ToCascade.append(color_filter_from_greyscale_filter(make_blur_filter(int(simpledialog.askstring(title="Blur",
                                                              prompt="Please enter kernel size for blur filter:")))))
            elif FILTER == 'sharpen':
                 ToCascade.append(color_filter_from_greyscale_filter(make_sharpen_filter(int(simpledialog.askstring(title="Sharpen",
                                                               prompt="Please enter kernel size for sharpen filter:")))))   
            elif FILTER == 'seam_carving':
                ToCascade.append(seam_maker(int(simpledialog.askstring(title="Seam Carving",
                                                              prompt="Please enter how many columns you want to remove for seam carving filter:"))))   
            elif FILTER == 'invert':
                ToCascade.append(color_filter_from_greyscale_filter(inverted))
                
            elif FILTER == 'detect_edges':
                ToCascade.append(color_filter_from_greyscale_filter(edges))  
                
            else:
                messagebox.showinfo("invalid Filter", f"The Filter you inputted, {FILTER}, is invalid, please start over.")
                
        TotalFilter = filter_cascade(ToCascade)
        Im = TotalFilter(Im)
        save_color_image(Im, "newImage.png")
            
        messagebox.showinfo("Succes!", "Your filtered image has been created in the same location as where you stored this software")
                
    elif USER_INP2 == 'BW' or USER_INP2 == 'bw':
        USER_INP3 = simpledialog.askstring(title="Filters",
                                  prompt="Please choose filter to apply to your image, choose from invert,blur,sharpen,and detect_edges, for a cascaded filter effect, please input multiple filters with comma and no space between:")
        Im = load_greyscale_image(USER_INP)

        Filters = USER_INP3.split(",")
        
        ToCascade = []
        
        for FILTER in Filters:
            if FILTER == 'blur':
                ToCascade.append(make_blur_filter(int(simpledialog.askstring(title="Blur",
                                                              prompt="Please enter kernel size for blur filter:"))))
            elif FILTER == 'sharpen':
                 ToCascade.append(make_sharpen_filter(int(simpledialog.askstring(title="Sharpen",
                                                               prompt="Please enter kernel size for sharpen filter:"))))   
            elif FILTER == 'invert':
                ToCascade.append(inverted)
                
            elif FILTER == 'detect_edges':
                ToCascade.append(edges)  
                
            else:
                messagebox.showinfo("invalid Filter", f"The Filter you inputted, {FILTER}, is invalid, please start over.")
        
        TotalFilter = filter_cascade(ToCascade)
        Im = TotalFilter(Im)
        save_greyscale_image(Im, "newImage.png")
            
        messagebox.showinfo("Succes!", "Your filtered image has been created in the same location as where you stored this software!")

    else:
        messagebox.showinfo("Invalid Color Mode", f"The color mode you inputted, {USER_INP2}, is invalid, please start over.")
        
    pass