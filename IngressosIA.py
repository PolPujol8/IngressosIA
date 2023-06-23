import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

sales_df = pd.read_csv("datos_de_ventas.csv")

#Visualització
sns.scatterplot(data=sales_df, x='Temperature', y='Revenue')
#plt.show()

#Creació Entrenament
x_train = sales_df['Temperature']
y_train = sales_df['Revenue']

#Creació model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape = [1]))

#Compilació
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

#Entrenament
epochs_hist = model.fit(x_train, y_train, epochs = 1000)

keys = epochs_hist.history.keys()

#Gràfic entrenament 
plt.plot(epochs_hist.history['loss'])
plt.title("Progres perdida")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.legend("Training loss")
plt.show()

weights = model.get_weights()

#Predicció
Temp = 50
Revenue = model.predict([Temp])
print("El benefici segons la red neuronal es de: ", Revenue)

#Gràfic de predicció
plt.scatter(x_train, y_train, color='gray')
plt.plot(x_train, model.predict(x_train), color='red')
plt.ylabel('Benefici [Dolars]')
plt.xlabel('Temperatura [Cº]')
plt.title('Ganancia VS Temperatura @Empresa Gelats')
plt.show()





