from dataloader import train_loader

images, labels = next(
    iter(train_loader)
)

print(images.shape)
print(labels.shape)
print(labels)