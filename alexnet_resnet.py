import torch
import urllib
import glob
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import wget
import collections
import time
DEBUG_MODE=False
# sample execution (requires torchvision)
def deeplearning(filename, modelname, start_image, end_image):
    start_image = str(start_image)
    end_image = str(end_image)
    inference_result = []
    time_start = time.time()
    if modelname == "alexnet":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        model.eval()
    else:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model.eval()

    file_list = glob.glob('./' + filename)
    probabilities = torch.Tensor()
    current_image = start_image
    # get "imagenet_classes.txt" ,if no local file, get from internet
    import os.path
    if os.path.exists("imagenet_classes.txt"):
        pass
    else:
        import urllib
        url, filename = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
        try:
            urllib.URLopener().retrieve(url, filename)
        except:
            urllib.request.urlretrieve(url, filename)

    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    if DEBUG_MODE:
        print("start inference for: "+modelname)
        print("inference image from "+start_image+" to "+end_image)
    for folder in file_list:

        while int(current_image) <= int(end_image):
            for f in glob.glob(folder + '/test_' + str(current_image) + '.JPEG'):
                input_image = Image.open(f)
                if input_image.mode != 'RGB':
                    input_image = input_image.convert("RGB")
                    os.remove(f)
                    input_image.save(f)

                try:
                    preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                except:
                    current_image = int(current_image)+1
                    break
                input_tensor = preprocess(input_image)
                input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

                # move the input and model to GPU for speed if available
                if torch.cuda.is_available():
                    input_batch = input_batch.to('cuda')
                    model.to('cuda')

                with torch.no_grad():
                    output = model(input_batch)
                # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
                # print(output[0])

                # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                # print(probabilities)

                # Show top categories per image
                top_prob, top_catid = torch.topk(probabilities, min(1, probabilities.shape[0]))
                for i in range(top_prob.size(0)):
                    image_name = 'test_' + str(current_image) + '.JPEG'
                    inference_result.append((image_name,categories[top_catid[i]],top_prob[i].item()))
                    # print(categories[top_catid[i]], top_prob[i].item())

            current_image = int(current_image)+1
    time_end=time.time()
    return inference_result,(time_end-time_start)
