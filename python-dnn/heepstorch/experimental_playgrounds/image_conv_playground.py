import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt


def im2col(input_data, kernel_height, kernel_width):
    raise ValueError()


def conv2d_im2col(input_data, kernel):
    raise ValueError()


# Load and prepare image
dataset = datasets.CIFAR10(root='./data', train=True, download=True)
image = dataset[1][0]
transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
img_tensor = transform(image).unsqueeze(0)

# Sobel kernels
sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

# Compare methods
edge_x_torch = F.conv2d(img_tensor, sobel_x)
edge_y_torch = F.conv2d(img_tensor, sobel_y)
magnitude_torch = torch.sqrt(edge_x_torch ** 2 + edge_y_torch ** 2)

edge_x_im2col = conv2d_im2col(img_tensor, sobel_x)
edge_y_im2col = conv2d_im2col(img_tensor, sobel_y)
magnitude_im2col = torch.sqrt(edge_x_im2col ** 2 + edge_y_im2col ** 2)

# Compare results
diff = torch.abs(magnitude_torch - magnitude_im2col).max()
print(f"Max difference between methods: {diff:.10f}")

# Plot results
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(image)
plt.title('Original')
plt.subplot(132)
plt.imshow(magnitude_torch[0, 0].detach(), cmap='gray')
plt.title('Magnitude (F.conv2d)')
plt.subplot(133)
plt.imshow(magnitude_im2col[0, 0].detach(), cmap='gray')
plt.title('Magnitude (im2col)')
plt.show()
