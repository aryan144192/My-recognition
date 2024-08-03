#!/usr/bin/python3                # This just tells the computer which Python to use
import jetson_inference           # Jetson Inference is a libaray where we can acess the AI models
import jetson_utils               # Jetson Untils has fuctions I can use to train and acess the AI and loading images
import argparse                   

parser=argparse.ArgumentParser()  # This creates an object to parse the terminal for specific stuff
parser.add_argument("filename", type=str, help="filename of the image to processes") # This gathers the data from the terminal suchas filename
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be: googlenet, resnet-18") # This gathers the --network and defaults to googlenet if not specfied
opt = parser.parse_args() # This parses the terminal for the filename

img = jetson_utils.loadImage(opt.filename) # This loads the image for the model to analyze
net = jetson_inference.imageNet(opt.network) # This loads the network used

class_idx, confidence = net.Classify(img) # this returns the output of the class the model recognized and the conficence
class_desc = net.GetClassDesc(class_idx) # this changes the output into normal english so we can understand

print(f"images recognized as {class_desc} (class# {class_idx}) with {confidence*100}% confidence") # This prints our output
