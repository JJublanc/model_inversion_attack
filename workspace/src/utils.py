import matplotlib.pyplot as plt
import numpy as np
from numpy import expand_dims
from tensorflow.keras.datasets.mnist import load_data
from numpy.random import randint


def load_mnist_data(data_type):
    
    if data_type=="private":
        classes = [0,1,2,3,4]
    elif data_type=="public":
        classes = [5,6,7,8,9]
    elif data_type=="all":
        classes = [0,1,2,3,4,5,6,7,8,9]
    
    # load dataset
    (train_X, train_y), (test_X, test_y) = load_data()
    
    # select train images and labels for the given classes
    selected_train_ix = [(x in classes) for x in train_y]
    train_X = train_X[selected_train_ix]
    train_y = train_y[selected_train_ix]

    # select test images and labels for the given classes
    selected_test_ix = [(x in classes) for x in test_y]
    test_X = test_X[selected_test_ix]
    test_y = test_y[selected_test_ix]
    
    # expand to 3d, e.g. add channels and convert to float32
    train_X = expand_dims(train_X, axis=-1).astype('float32')
    test_X = expand_dims(test_X, axis=-1).astype('float32')
    
    # scale from [0,255] to [-1,1]
    train_X = train_X / 255
    test_X = test_X / 255
    
    print("train size : {}".format(len(train_X)))
    print("test size : {}".format(len(test_X)))
    print("total size : {}".format(len(train_X) + len(test_X)))
    
    return train_X, train_y, test_X, test_y

def plot_img(data, labels, dim=(5, 5), im_shape=(28,28), fig_size=(10,10)):
    
    nb_images = dim[0]*dim[1]
    figsize=fig_size
    plt.figure(figsize=figsize)
    random_index = np.random.randint(0, len(data), nb_images)
    
    for i in range(nb_images):
        index = random_index[i]
        img = np.reshape(data[index],im_shape)
        
        plt.subplot(dim[0], dim[1], i +1)
        plt.imshow(img, cmap="gray")
        plt.title("label : {}".format(labels[index]))
        plt.axis('off')
        
def pick_and_show_image(data, labels):
    i = np.random.randint(0,len(data),1)[0]
    img = data[i][:,:,0]
    y = labels[i]
    
    figsize=(3,3)
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap="gray")
    plt.title("Label : {}".format(y))
    plt.axis('off')
    plt.show()
    return img, y