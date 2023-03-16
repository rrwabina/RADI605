# from torchvision.utils import make_grid

# class_names = ['Without', 'With']
# for images, labels in train_loader: 
#     break

# print('Label:', labels.numpy())
# print('Class: ', *np.array([class_names[i] for i in labels]))

# im = make_grid(images, nrow = 6)  
# plt.figure(figsize=(15, 8))
# plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))


# def imshow(inp, title = None):
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  


# inputs, classes = next(iter(train_loader))
# out = torchvision.utils.make_grid(inputs)
# imshow(out)



# self.augment = {
#                     'train': transforms.Compose([
#                         transforms.Resize(50),
#                         transforms.RandomResizedCrop(224),
#                         transforms.ToTensor(),
#                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                     ]),
#                     'val':   transforms.Compose([
#                         transforms.Resize(50),
#                         transforms.CenterCrop(224),
#                         transforms.ToTensor(),
#                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                     ]),
#                 }