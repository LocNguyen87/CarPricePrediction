import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pickle

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

# filling in missing values
for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())

# making data a numpy array like
x = data.drop(['price'], axis=1)
y = data.price
x = x.values
y = y.values

# dividing data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)

# standardzing the data
stds = StandardScaler()
scaler = stds.fit(x_train)
rescaledx = scaler.transform(x_train)

# selecting and fitting the model for training
model = DecisionTreeRegressor()
model.fit(rescaledx, y_train)

# saving the trained mode
pickle.dump(model, open('rf_model.pkl', 'wb'))
# saving StandardScaler
pickle.dump(stds, open('scaler.pkl', 'wb'))