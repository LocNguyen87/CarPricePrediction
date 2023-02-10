import streamlit as st
import pickle
import pandas as pd


def main():
    df = pd.read_csv('./carprice.csv')
    style = """<div>
              <h1 style='color:blue'>Car Price Prediction</h1>
       </div>"""
    st.markdown(style, unsafe_allow_html=True)
    st.dataframe(df)
    col1, col2, col3, col4 = st.columns(4)

    # combobox mapping data
    fueltype_lbl = ('gas','diesel')
    fueltype_val = (1,0)
    fueltype_data = dict(zip(fueltype_lbl, fueltype_val))
    aspiration_lbl  = ('std','turbo')
    aspiration_val = (0,1)
    aspiration_data = dict(zip(aspiration_lbl, aspiration_val))
    doornumber_lbl  = ('two','four')
    doornumber_val = (1,0)
    doornumber_data = dict(zip(doornumber_lbl, doornumber_val))
    carbody_lbl = ('convertible','hatchback','sedan','wagon','hardtop')
    carbody_val = (0, 2, 3, 4, 1)
    carbody_data = dict(zip(carbody_lbl, carbody_val))
    drivewheel_lbl = ('rwd','fwd','4wd')
    drivewheel_val = (2, 1, 0)
    drivewheel_data = dict(zip(drivewheel_lbl, drivewheel_val))
    enginelocation_lbl = ('front','rear')
    enginelocation_val = (0, 1)
    enginelocation_data = dict(zip(enginelocation_lbl, enginelocation_val))
    enginetype_lbl = ('dohc','ohcv','ohc','l','rotor','ohcf','dohcv')
    enginetype_val = (0, 5, 3, 2, 6, 4, 1)
    enginetype_data = dict(zip(enginetype_lbl, enginetype_val))
    cylindernumber_lbl = ('four','six','five','three','twelve','two','eight')
    cylindernumber_val = (2, 3, 1, 4, 5, 6, 0)
    cylindernumber_data = dict(zip(cylindernumber_lbl, cylindernumber_val))
    fuelsystem_lbl = ('mpfi','2bbl', 'mfi','1bbl','spfi','4bbl','idi','spdi')
    fuelsystem_val = (5, 1, 4, 0, 7, 2, 3, 6)
    fuelsystem_data = dict(zip(fuelsystem_lbl, fuelsystem_val))
    carbrand_lbl = ('alfa-romero','audi','bmw','chevrolet','dodge','honda','isuzu','jaguar','mazda','buick','mercury','mitsubishi','nissan','peugeot','plymouth','porsche','renault','saab','subaru','toyota','volkswagen','volvo')
    carbrand_val = (0, 1, 2, 4, 5, 6, 7, 8, 9, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
    carbrand_data = dict(zip(carbrand_lbl, carbrand_val))
    carmodel_lbl = ('giulia', 'stelvio', 'quadrifoglio', '100ls', 'fox', '5000','4000','5000s(diesel)','320i','x1','x3','z4','x4','x5','impala','montecarlo','vega2300','rampage','challengerse','d200','monaco(sw)','colthardtop','colt(sw)','coronetcustom','dartcustom','coronetcustom(sw)','civic','civiccvcc','accordcvcc','accordlx','civic1500gl','accord','civic1300','prelude','civic(auto)','mu-x','d-max','d-maxv-cross','xj','xf','xk','rx3','glcdeluxe','rx2coupe','rx-4','626','glc','rx-7gs','glc4','glccustoml','glccustom','electra225custom','centuryluxus(sw)','century','skyhawk','opelisuzudeluxe','skylark','centuryspecial','regalsportcoupe(turbo)','cougar','mirage','lancer','outlander','g4','mirageg4','montero','pajero','versa','gt-r','rogue','latio','titan','leaf','juke','note','clipper','nv200','dayz','fuga','otti','teana','kicks','504', '304','504(sw)','604sl','505sturbodiesel','furyiii','cricket','satellitecustom(sw)','furygransedan','valiant','duster','macan','panamera', 'cayenne','boxter','12tl','5gtl','99e','99le','99gle','','dl','brz','baja','r1','r2','trezia','tribeca','coronamarkii','corona','corolla1200','coronahardtop','corolla1600(sw)','carina','markii','corolla','corollaliftback','celicagtliftback','corollatercel','coronaliftback','starlet','tercel','cressida','celicagt','rabbit','1131deluxesedan','model111','type3','411(sw)','superbeetle','dasher','rabbitcustom','145e(sw)','144ea','244dl','245','264gl','diesel','246')
    carmodel_val = (76,122,106,1,71,14,12,15,11,133,134,140,135,136,83,95,131,111,38,63,94,46,45,56,64,57,39,43,26,27,42,25,41,105,40,97,61,62,138,137,139,117,81,116,114,21,77,115,78,80,79,70,36,35,119,100,120,37,112,58,91,86,102,75,92,96,103,132,82,113,87,126,88,84,98,44,99,66,72,101,124,85,16,10,17,20,18,74,60,118,73,130,69,89,104,32,29,3,19,22,24,23,0,68,30,28,107,108,127,128,55,52,48,53,49,31,90,47,50,34,51,54,121,125,59,33,109,2,93,129,13,123,65,110,5,4,6,7,9,67,8)
    carmodel_data = dict(zip(carmodel_lbl, carmodel_val))
    # end combobox mapping data

    # form interface
    carbrand_sl = col1.selectbox('Car Brand', carbrand_lbl, index=carbrand_lbl.index('volvo'))
    carmodel_sl = col2.selectbox('Car Model', carmodel_lbl, index=carmodel_lbl.index('diesel'))
    symboling = col3.number_input('Symboling', step=1.0, format='%.1f', value=-1.0)
    fueltype_sl = col4.selectbox('Fuel Type', fueltype_lbl, index=fueltype_lbl.index('gas'))
    aspiration_sl = col1.selectbox('Aspiration', aspiration_lbl, index=aspiration_lbl.index('turbo'))
    doornumber_sl = col2.selectbox('Door Number', doornumber_lbl, index=doornumber_lbl.index('four'))
    carbody_sl = col3.selectbox('Car Body', carbody_lbl, index=carbody_lbl.index('wagon'))
    drivewheel_sl = col4.selectbox('Drive Wheel :white_check_mark:', drivewheel_lbl, index=drivewheel_lbl.index('rwd'))
    enginelocation_sl = col1.selectbox('Engine Location', enginelocation_lbl, index=enginelocation_lbl.index('front'))
    wheelbase = col2.number_input('Wheel Base :white_check_mark:',  step=1.0, format='%.1f', value=104.3)
    carlength = col3.number_input('Car Length :white_check_mark:',  step=1.0, format='%.1f', value=188.8)
    carwidth = col4.number_input('Car Width :white_check_mark:',  step=1.0, format='%.1f', value=67.2)
    carheight = col1.number_input('Car Height',  step=1.0, format='%.1f', value=57.5)
    curbweight  = col2.number_input('Curb Weight :white_check_mark:',  step=1.0, format='%.1f', value=3157.0)
    enginetype_sl = col3.selectbox('Engine Type', enginetype_lbl, index=enginetype_lbl.index('ohc'))
    cylindernumber_sl = col4.selectbox('Cylinder Number', cylindernumber_lbl, index=cylindernumber_lbl.index('four'))
    enginesize = col1.number_input('Engine Size :white_check_mark:',  step=1.0, format='%.1f', value=130.0)
    fuelsystem_sl = col2.selectbox('Fuel System :white_check_mark:', fuelsystem_lbl, index=fuelsystem_lbl.index('mpfi'))
    boreratio = col3.number_input('Bore Ratio :white_check_mark:',  step=1.0, format='%.2f', value=3.62)
    stroke = col4.number_input('Stroke',  step=1.0, format='%.2f', value=3.15)
    compressionratio = col1.number_input('compressionratio',  step=1.0, format='%.1f', value=7.5)
    horsepower = col2.number_input('Horse Power :white_check_mark:',  step=1.0, format='%.1f', value=162.0)
    peakrpm = col3.number_input('peakrpm',  step=1.0, format='%.1f', value=5100.0)
    citympg = col4.number_input('City Mpg :white_check_mark:',  step=1.0, format='%.1f', value=17.0)
    highwaympg = col1.number_input('Highway Mpg :white_check_mark:',  step=1.0, format='%.1f', value=22.0)

    button = st.button('Predict')
    # if button is pressed
    if button:
        # transform combobox data
        fueltype = fueltype_data[fueltype_sl]
        aspiration = aspiration_data[aspiration_sl]
        doornumber = doornumber_data[doornumber_sl]
        carbody = carbody_data[carbody_sl]
        drivewheel = drivewheel_data[drivewheel_sl]
        enginelocation = enginelocation_data[enginelocation_sl]
        enginetype = enginetype_data[enginetype_sl]
        cylindernumber = cylindernumber_data[cylindernumber_sl]
        fuelsystem = fuelsystem_data[fuelsystem_sl]
        carbrand = carbrand_data[carbrand_sl]
        carmodel = carmodel_data[carmodel_sl]
        # make prediction
        result = predict(carlength, carwidth, curbweight, enginesize, horsepower, wheelbase, boreratio, citympg, highwaympg, drivewheel, fuelsystem)
        st.success(f'The value of the car is ${result}')

# load the train model
with open('rf_model.pkl', 'rb') as rf:
    model = pickle.load(rf)

# load the StandardScaler
with open('scaler.pkl', 'rb') as stds:
    scaler = pickle.load(stds)

def predict(carlength, carwidth, curbweight, enginesize, horsepower, wheelbase, boreratio, citympg, highwaympg, drivewheel, fuelsystem):
    # create data frame
    lists = [carlength, carwidth, curbweight, enginesize, horsepower, wheelbase, boreratio, citympg, highwaympg, drivewheel, fuelsystem]
    df = pd.DataFrame(lists).values.reshape(-1,11)
    # scaling the data
    dfs = scaler.transform(df)
    # making predictions using the train model
    prediction = model.predict(dfs)
    result = int(prediction)
    return result

if __name__ == '__main__':
    main()