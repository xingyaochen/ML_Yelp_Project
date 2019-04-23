from PIL import Image, ImageDraw
import os


def makeGif(folder, duration, gifName):
    imgFiles = sorted(os.listdir(folder))
    images = [Image.open(folder+ "/" + fn) for fn in imgFiles]
    images[0].save(gifName, format='GIF', append_images=images[1:], save_all=True, duration=duration, loop=0)


# folders = os.listdir('earningMaps/')
# for folder in folders:
#     gifName = folder+"_earning.gif"
#     folder = "earningMaps/"+folder 
#     print(folder)
#     try:
#         makeGif(folder, 400, gifName)
#     except:
#         pass
