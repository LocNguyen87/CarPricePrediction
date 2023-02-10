import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import pickle
from matplotlib import pyplot as plt

data = pd.read_csv('./carprice.csv')

# split the CarName column to car brand and car model columns
data['carbrand'] = data['CarName'].apply(lambda x: x.split(" ")[0].lower())
data['carmodel'] = data['CarName'].apply(lambda x: "".join(x.split(" ")[1:]).lower())

# fix car brand typo
data.carbrand.replace(to_replace='toyouta', value='toyota', inplace=True)
data.carbrand.replace(to_replace=['vokswagen', 'vw'], value='volkswagen', inplace=True)
data.carbrand.replace(to_replace=['maxda'], value='mazda', inplace=True)
data.carbrand.replace(to_replace=['porcshce'], value='porsche', inplace=True)

# remove not needed columns
data.drop(columns=['car_ID', 'CarName'], inplace=True)

# encode object columns
label_encoder = LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])
    
# making data a numpy array like
# x = data.drop(['price'], axis=1)
x = data.loc[:, ['carlength', 'carwidth', 'curbweight', 'enginesize', 'horsepower', 'wheelbase', 'boreratio', 'citympg', 'highwaympg', 'drivewheel', 'fuelsystem']]
y = data.price
x = x.values
y = y.values

# dividing data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

# standardzing the data
stds = StandardScaler()
scaler = stds.fit(x_train)
rescaledx = scaler.transform(x_train)

# selecting and fitting the model for training
model = RandomForestRegressor()
model.fit(rescaledx, y_train)

x_test_scaled = scaler.transform(x_test)
y_pred = model.predict(x_test_scaled)
print(r2_score(y_test, y_pred))
print(y_test)
print(y_pred)

# plt.scatter(y_test, y_pred, c=['blue','red'])
# plt.xlabel('Actual Price')
# plt.ylabel('Predict Price')
# plt.show()

# plot for pred/test comparison
plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred, c='crimson')

p1 = max(max(y_pred), max(y_test))
p2 = min(min(y_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

# # saving the trained mode
# pickle.dump(model, open('rf_model.pkl', 'wb'))
# # saving StandardScaler
# pickle.dump(stds, open('scaler.pkl', 'wb'))