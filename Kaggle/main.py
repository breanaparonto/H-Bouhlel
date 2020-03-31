import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

from mnist_model import one_hot_encode, model_mnist

train_data = pd.read_csv(r"C:\Users\ahmed\Desktop\M2_BD\2e semeste\H-Bouhlel\Kaggle/train.csv")
test_data = pd.read_csv(r"C:\Users\ahmed\Desktop\M2_BD\2e semeste\H-Bouhlel\Kaggle/test.csv")

y_train = train_data['label'].values
X_train = train_data.drop('label', axis=1).values

y_train = one_hot_encode(y_train)

X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

model=model_mnist()
model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_training, y_training, epochs=25, batch_size=64,verbose=2,validation_data=(X_valid, y_valid))

results = model.predict(test_data)
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("mnist.csv",index=False)