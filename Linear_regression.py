#linear Regression , ! variable 


'''
import numpy as np
X = np.array([1,2,3,4])
Y = np.array([2,4,6,8])

b , w = 0 , 0
alpha = 0.1

for epoch in range(1000):
    y_pred = w*X + b

    dw = (-2/len(X)) * np.sum(X *(Y - y_pred))
    db = (-2/len(X)) * np.sum(Y - y_pred)

    w = w - (alpha * dw)
    b = b - (alpha * db) 

print(w,b)

'''

# Polynomial regression
import numpy as np
X= np.array([1,2,3,4])
Y = np.array([2,4,6,8])

b , w = 0 , 0
alpha = 0.1

