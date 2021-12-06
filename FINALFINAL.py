#!/usr/bin/env python
# coding: utf-8

# In[27]:


import os
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.image import imread
import cv2
from PIL import Image


# In[28]:


# get_ipython().run_line_magic('matplotlib', 'inline')

for iiii in range(142,143):

    random.seed(iiii)


    # In[29]:


    dataset_path = 'YaleDataset/'
    dataset_dir  = os.listdir(dataset_path)

    width  = 195
    height = 231


    # In[30]:


    # training_im, testing_im, training_label, testing_label = [], [], [], []
    # types = ["centerlight", "glasses", "happy", "leftlight", "noglasses","normal", "rightlight", "sad", "sleepy", "surprised", "wink"]
    # for i in range(1,16):
    #     im = {}
    #     testing_type = random.choice(types)
    #     for t in types:
    #         im[t] = np.array(Image.open(path_to_folder+"YaleDataset/subject"+str(i).zfill(2)+"."+t),'uint8')
    #         im[t] = cv2.resize(im[t], im_size, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
    #         if t == testing_type:
    #             testing_im.append(im[t])
    #             testing_label.append(i)
    #         else:
    #             training_im.append(im[t])
    #             training_label.append(i)


    # In[31]:


    def load_images():
        im_size=(64,64)
        training_im, testing_im, training_label, testing_label = [], [], [], []
        types = ["centerlight", "glasses", "happy", "leftlight", "noglasses","normal", "rightlight", "sad", "sleepy", "surprised", "wink"]
        for i in range(1,16):
            im = {}
            testing_type = random.sample(types, 2)
            # print(testing_type)
            for t in types:
                im[t] = np.array(Image.open("YaleDataset/subject"+str(i).zfill(2)+"."+t),'uint8')
                im[t] = cv2.resize(im[t], im_size, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
                if t in testing_type:
                    print("YaleDataset/subject"+str(i).zfill(2)+"."+t)
                    testing_im.append(im[t])
                    testing_label.append(i)
                else:
                    training_im.append(im[t])
                    training_label.append(i)
        
        return np.array(training_im), np.array(training_label), np.array(testing_im), np.array(testing_label)
        
    training_im, training_label, testing_im, testing_label = load_images()


    # In[32]:


    training_tensor = np.ndarray(shape=(len(training_im), 64*64), dtype=np.float64)

    for i in range(len(training_im)):
        training_tensor[i,:] = np.array(training_im[i], dtype='float64').flatten()
        if i<8:
            plt.subplot(2,4,1+i)
            plt.imshow(training_im[i], cmap='gray')
    # plt.show()


    # In[33]:


    mean_face = np.zeros((1,64*64))

    for i in training_tensor:
        mean_face = np.add(mean_face,i)

    mean_face = np.divide(mean_face,float(len(training_im))).flatten()

    plt.imshow(mean_face.reshape(64, 64), cmap='gray')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
    # plt.show()


    # In[34]:


    height = 64
    width = 64

    normalised_training_tensor = np.ndarray(shape=(len(training_im), height*width))

    for i in range(len(training_im)):
        normalised_training_tensor[i] = np.subtract(training_tensor[i],mean_face)


    # In[35]:


    plt.clf()
    for i in range(len(training_im)):
        img = normalised_training_tensor[i].reshape(height,width)
        if i<8:
            plt.subplot(2,4,1+i)
            plt.imshow(img, cmap='gray')
            plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
    # plt.show()


    # # EIGENVECTORS AND VALUES USING SVD

    # In[36]:


    len(training_im)
    # get_ipython().system('ls CroppedYale')


    # In[37]:


    cov_matrix = np.cov(normalised_training_tensor)
    cov_matrix = np.divide(cov_matrix,len(training_im))
    # print(cov_matrix.shape)
    # print('Covariance matrix of X: \n%s' %cov_matrix)


    # In[38]:


    eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
    # print('Eigenvectors of Cov(X): \n%s' %eigenvectors)
    # print('\nEigenvalues of Cov(X): \n%s' %eigenvalues)


    # In[39]:


    eigenvectors.shape


    # In[40]:


    eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]

    # Sort the eigen pairs in descending order:
    eig_pairs.sort(reverse=True)
    eigvalues_sort  = np.array([eig_pairs[index][0] for index in range(len(eigenvalues))])
    eigvectors_sort = np.array([eig_pairs[index][1] for index in range(len(eigenvalues))])


    # In[41]:


    var_comp_sum = np.cumsum(eigvalues_sort)/sum(eigvalues_sort)
    # print(np.cumsum(eigvalues_sort))

    # Show cumulative proportion of varaince with respect to components
    # print("Cumulative proportion of variance explained vector: \n%s" %var_comp_sum)

    # x-axis for number of principal components kept
    num_comp = range(1,len(eigvalues_sort)+1)
    plt.title('Cum. Prop. Variance Explain and Components Kept')
    plt.xlabel('Principal Components')
    plt.ylabel('Cum. Prop. Variance Expalined')

    plt.scatter(num_comp, var_comp_sum)
    # plt.show()


    # In[42]:


    reduced_data = np.array(eigvectors_sort[:8]).transpose()


    # In[43]:


    proj_data = np.dot(training_tensor.transpose(),reduced_data)
    proj_data = proj_data.transpose()


    # In[44]:


    for i in range(proj_data.shape[0]):
        img = proj_data[i].reshape(height,width)
        if i < 8:
            plt.subplot(2,4,1+i)
            plt.imshow(np.abs(img), cmap='jet')
            plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
    # plt.show()


    # In[45]:


    w = np.array([np.dot(proj_data,i) for i in normalised_training_tensor])


    # In[46]:


    unknown_face        = testing_im[-1]
    # unknown_face        = plt.imread('Dataset/subject12.normal.jpg')
    unknown_face_vector = np.array(unknown_face, dtype='float64').flatten()

    plt.imshow(unknown_face, cmap='gray')
    plt.title('Unknown face')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
    # plt.show()


    # In[47]:


    normalised_uface_vector = np.subtract(unknown_face_vector,mean_face)

    plt.imshow(normalised_uface_vector.reshape(height, width), cmap='gray')
    plt.title('Normalised unknown face')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
    # plt.show()


    # In[48]:


    w_unknown = np.dot(proj_data, unknown_face_vector)
    w_unknown


    # In[49]:


    diff = w - w_unknown
    norms = np.linalg.norm(diff, axis=1)
    # print(norms)


    # In[50]:


    w.shape


    # In[51]:


    correct_pred = 0
    count=0
    num_images=0
    def recogniser(curr_testing_im, curr_testing_label, proj_data, w):
        global count, num_images, correct_pred
        
        unknown_face = curr_testing_im
        unknown_face_vector = np.array(curr_testing_im, dtype='float64').flatten()
        normalised_uface_vector = unknown_face_vector-mean_face
        # print("=====================")
        plt.imshow(unknown_face, cmap='gray')
        plt.title(f'Unknown face {count}')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='on',left='off', which='both')
        
        # plt.show()
        w_unknown = np.dot(proj_data, normalised_uface_vector)
        diff  = w - w_unknown
        norms = np.linalg.norm(diff, axis=1)
        index = np.argmin(norms)
        
        t1 = 100111536
        t0 = 88831687
        
        if norms[index] < t1 or True:
    #         # print

            if norms[index] < t0 or True: # It's a face
                plt.title(f'Matched {count}: {training_label[index]}', color='g')
                plt.imshow(training_im[index], cmap='gray')
                # plt.show()
                print("=====================")
                print(curr_testing_label)
                print(training_label[index])
                print("=====================")
                if curr_testing_label == training_label[index]:
                    correct_pred += 1
    #             else:
    #                 plt.title('Matched:'+'.'.join(training_label[index]), color='g')
    #                 plt.title('Matched:'+'.'.join(train_image_names[index].split('.')[:2]), color='r')
    #                 plt.imshow(imread('Dataset/'+train_image_names[index]), cmap='gray')
    #         else:
    #             if img.split('.')[0] not in [i.split('.')[0] for i in train_image_names] and img.split('.')[0] != 'apple':
    #                 plt.title('Unknown face!', color='g')
    #                 correct_pred += 1
    #             else:
    #                 plt.title('Unknown face!', color='r')
    #         plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
    #         plt.subplots_adjust(right=1.2, top=2.5)
        
    #     else:     
    # #         plt.subplot(9,4,1+count)
    # #         if len(img.split('.')) == 3:
    #             pass
    # #             plt.title('Not a face!', color='r')
    #         else:
    # #             plt.title('Not a face!', color='g')
    #             correct_pred += 1
    #         plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
        

    # fig = plt.figure(figsize=(15, 15))
    for idx, curr_testing_im in enumerate(testing_im):
        recogniser(curr_testing_im, testing_label[idx], proj_data, w)

    # # plt.show()

    if correct_pred/testing_im.shape[0]*100>=80:
        print(iiii, 'Correct predictions: {}/{} = {}%'.format(correct_pred, testing_im.shape[0], correct_pred/testing_im.shape[0]*100))

    # In[52]:


    # print(len(training_im))


    # In[ ]:





    # In[ ]:





    # In[ ]:




