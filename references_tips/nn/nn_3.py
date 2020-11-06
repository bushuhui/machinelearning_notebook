%matplotlib nbagg

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
imgs = []

y_true = np.array(nn.y).astype(float)

# back-propagation
def backpropagation(n, X, y):
    for i in range(n.n_epoch):
        # forward to calculate each node's output
        forward(n, X)
        
        # print loss, accuracy
        L = np.sum((n.z2 - y)**2)
        
        y_pred = np.zeros(nn.z2.shape[0])
        y_pred[np.where(nn.z2[:,0]<nn.z2[:,1])] = 1
        acc = accuracy_score(y_true, y_pred)
        
        print("epoch [%4d] L = %f, acc = %f" % (i, L, acc))
        
        # calc weights update
        d2 = n.z2*(1-n.z2)*(y - n.z2)
        d1 = n.z1*(1-n.z1)*(np.dot(d2, n.W2.T))
        
        # update weights
        n.W2 += n.epsilon * np.dot(n.z1.T, d2)
        n.b2 += n.epsilon * np.sum(d2, axis=0)
        n.W1 += n.epsilon * np.dot(X.T, d1)
        n.b1 += n.epsilon * np.sum(d1, axis=0)
        
        # plot animation
        #img = plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=plt.cm.Spectral)
        #imgs.append(img)

nn.n_epoch = 2000
backpropagation(nn, X, t)

#ani = animation.ArtistAnimation(fig, imgs)
#plt.show()
