"""
Model's inference
* Just replace the path to your current directoty with mine and the rest of the code should work!
* You also need to modify the path_to_model; I uploaded the model weights to the course box.
"""

from PIL import Image

output_dir = '/content/dataset'
current_directory = os.path.join('/content'+ output_dir, 'test') # Replace this with yours!
path_to_model = '/content/drive/MyDrive/UIC/checkpoint/model.pth' # This is the google drive path, you may need to modify and set it locally!

transform = transform = transforms.Compose([
transforms.Resize((64, 64)),  # Downsampling to speed up processing
transforms.ToTensor(),
])

dev_dataset = datasets.ImageFolder(root = current_directory , transform = transform)
dev_loader  = DataLoader(dev_dataset, batch_size=25, shuffle = True)
# pick a random batch of size 25 of the testset, called it as our new devset!
X_dev,y_dev = next(iter(dev_loader))
X_dev = X_dev.float()

# Some printings!
print('Data shapes (train/test):')
print(X_dev.data.shape )

# and the range of pixel intensity values
print('\nData value range:')
print((torch.min(X_dev.data),torch.max(X_dev.data)) )

# Loading the model!
loaded_model,lossfun,optimizer = makeTheNet()
loaded_model.load_state_dict(torch.load(path_to_model,map_location=device))
loaded_model.eval()

with torch.no_grad():
        outputs = loaded_model(X_dev)
        _, predicted = torch.max(outputs, 1)

pic = X_dev.numpy().transpose((0,2,3,1)) # transpose the image in order to be plotted properly!

# Creating a 5 * 5 subplot!
fig,ax = plt.subplots(5,5,figsize=(15,15))
for i in range(25):
    ax[i//5,i%5].imshow(pic[i])
    ax[i//5,i%5].text(5,0,f"Image true label: {classes[y_dev[i].item()]}",ha='center',va='center',fontweight='bold',color='k',backgroundcolor='y', fontsize = 'x-small')
    if predicted[i].item() == y_dev[i].item():
        color = 'b'
    else:
        color = 'r'
    ax[i//5,i%5].text(5,5,f"Model's prediction: {classes[predicted[i].item()]}",ha='center',fontweight='bold',color=color,backgroundcolor='y', fontsize = 'x-small')
    ax[i//5,i%5].axis('off')


plt.tight_layout()
plt.show()
