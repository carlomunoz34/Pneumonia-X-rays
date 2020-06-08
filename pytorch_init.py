from pytorch_model import PneumoniaImages, VGG16
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

if __name__ == '__main__':
    print("Starting loading data in memory")
    image_size = 150
    train_dataset = PneumoniaImages(train=True,  image_size=image_size, small=True, add_contrast=True)
    test_dataset = PneumoniaImages(train=False,  image_size=image_size, small=True, add_contrast=True)
    print("Finished loading data in memory")
    
    train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=True)

    model = VGG16(image_size).cuda()
    print("Starting training")
    t0 = datetime.now()
    train_cost, train_acc, test_cost, test_acc = model.fit(train_loader, test_loader, epochs=5)
    t1 = datetime.now()

    print('Training time:', t1 - t0)

    plt.plot(train_cost, label='train cost')
    plt.plot(test_cost, label='test cost')
    plt.legend()
    plt.show()

    plt.plot(train_acc, label='train acc')
    plt.plot(test_acc, label='test_acc acc')
    plt.legend()
    plt.show()