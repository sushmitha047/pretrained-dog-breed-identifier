import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import __version__
import torch

resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)


def train_and_save_model(model_name, train_data, epochs=10):
    model = models[model_name]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        for inputs, labels in train_data:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Save the model weights
    torch.save(model.state_dict(), f'{model_name}_weights.pth')

models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

# obtain ImageNet labels
with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())

# def classifier(img_path, model_name):
#     # load the image
#     img_pil = Image.open(img_path)

#     # define transforms
#     preprocess = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     # preprocess the image
#     img_tensor = preprocess(img_pil)
    
#     # resize the tensor (add dimension for batch)
#     img_tensor.unsqueeze_(0)
    
#     # wrap input in variable, wrap input in variable - no longer needed for
#     # v 0.4 & higher code changed 04/26/2018 by Jennifer S. to handle PyTorch upgrade
#     pytorch_ver = __version__.split('.')
    
#     # pytorch versions 0.4 & hihger - Variable depreciated so that it returns
#     # a tensor. So to address tensor as output (not wrapper) and to mimic the 
#     # affect of setting volatile = True (because we are using pretrained models
#     # for inference) we can set requires_gradient to False. Here we just set 
#     # requires_grad_ to False on our tensor 
#     if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
#         img_tensor.requires_grad_(False)
    
#     # pytorch versions less than 0.4 - uses Variable because not-depreciated
#     else:
#         # apply model to input
#         # wrap input in variable
#         data = Variable(img_tensor, volatile = True) 

#     # apply model to input
#     model = models[model_name]

#     # puts model in evaluation mode
#     # instead of (default)training mode
#     model = model.eval()
    
#     # apply data to model - adjusted based upon version to account for 
#     # operating on a Tensor for version 0.4 & higher.
#     if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
#         output = model(img_tensor)

#     # pytorch versions less than 0.4
#     else:
#         # apply data to model
#         output = model(data)

#     # return index corresponding to predicted class
#     pred_idx = output.data.numpy().argmax()

#     return imagenet_classes_dict[pred_idx]

def classifier(img_path, model_name):
    # Load the model
    model = models[model_name]
    
    # Try to load saved weights
    try:
        model.load_state_dict(torch.load(f'{model_name}_weights.pth', weights_only=True))
        print(f"Loaded saved weights for {model_name}")
    except FileNotFoundError:
        print(f"No saved weights found for {model_name}, using pre-trained weights")
    
    # Put model in evaluation mode
    model.eval()
    
    # Load and preprocess the image
    img_pil = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img_pil)
    img_tensor.unsqueeze_(0)
    
    # Ensure we're not tracking gradients
    with torch.no_grad():
        output = model(img_tensor)
    
    # Get the predicted class
    pred_idx = output.argmax(dim=1).item()
    
    return imagenet_classes_dict[pred_idx]

