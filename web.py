import streamlit as st
from PIL import Image
import keras.utils as image
import numpy as np
import pandas as pd
import numpy as np 
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
 
# load model
model = load_model('model.h5')


st.write("""# This is the Breast Cancer Clasifier model used to clasify the cancer in the patient""")
select=st.radio("Select the model type to use",["image","textual"],index=0)
if select=="image":
           
    def load_image(img):
        im=Image.open(img)
        image=np.array(im)
        return image

    upload_file=st.file_uploader("Choose a image file")
    if upload_file is not None:
        img=load_image(upload_file)
        st.image(img)
        st.write("Image uploaded successfully")
    else:
        st.write("Image not uploaded")

    img=img
    img=np.resize(img,(32,32,3))
    # st.write(img.shape)

    img = image.img_to_array(img)
    img=np.resize(img,(1,32,32,3))
    img = img/255


    predict=model.predict(img)
    st.write(predict)
    predict.all()
    if predict[0][0]>0.5:
        st.write("The cancer has been clasified as benign type")
    else:
        st.write("The cancer has been clasified as malignent type")
else:
    
    
    
    def user_input_parameters():
        radius_measure=st.number_input("Enter the radius mean",min_value=0.0,max_value=50.0)
        texture_mean=st.number_input("Enter the Texture mean",min_value=0.0,max_value=50.0)
        perimeter_measure=st.number_input("Enter the perimeter mean",min_value=0.0,max_value=200.0)
        area_measure=st.number_input("Enter the area mean",min_value=0.0,max_value=2550.0)
        smoothness_measure=st.number_input("Enter the smoothness mean",min_value=0.0,max_value=5.0)
        compactness_measure=st.number_input("Enter the compactness mean",min_value=0.0,max_value=5.0)
        concave_measure=st.number_input("Enter the concave mean",min_value=0.0,max_value=5.0)
        concave_points_measure=st.number_input("Enter the concave_points mean",min_value=0.0,max_value=5.0)
        symmetry_measure=st.number_input("Enter the symmetry mean",min_value=0.0,max_value=5.0)
        fractal_diamension_measure=st.number_input("Enter the fractal_diamension mean",min_value=0.0,max_value=50.0)
        radius_seen_measure=st.number_input("Enter the radius_seen measure",min_value=0.0,max_value=50.0)
        texture_seen_measure=st.number_input("Enter the texture_seen measure",min_value=0.0,max_value=50.0)
        perimeter_seen_measure=st.number_input("Enter the perimeter_seen measure",min_value=0.0,max_value=50.0)
        area_seen_measure=st.number_input("Enter the area_seen measure",min_value=0.0,max_value=500.0)
        smoothness_seen_measure=st.number_input("Enter the smoothness_seen measure",min_value=0.0,max_value=5.0)
        compactness_seen_measure=st.number_input("Enter the compactness_seen measure",min_value=0.0,max_value=5.0)
        concavity_seen_measure=st.number_input("Enter the concaveity_seen measure",min_value=0.0,max_value=5.0)
        concave_points_seen_measure=st.number_input("Enter the concave_points_seen measure",min_value=0.0,max_value=10.0)
        symmetry_seen_measure=st.number_input("Enter the symmetry_seen measure",min_value=0.0,max_value=10.0)
        fractal_dimension_seen_measure=st.number_input("Enter the fractal_dimension_seen measure",min_value=0.0,max_value=5.0)
        radius_worst_measure=st.number_input("Enter the radius_worst measure",min_value=0.0,max_value=50.0)
        texture_worst_measure=st.number_input("Enter the texture_worst measure",min_value=0.0,max_value=50.0)
        perimeter_worst_measure=st.number_input("Enter the perimeter_worst measure",min_value=0.0,max_value=500.0)
        area_worst_measure=st.number_input("Enter the area_worst measure",min_value=0.0,max_value=500.0)
        smoothness_worst_measure=st.number_input("Enter the smoothness_worst measure",min_value=0.0,max_value=5.0)
        compactness_worst_measure=st.number_input("Enter the compactness_worst measure",min_value=0.0,max_value=5.0)
        
        concavity_worst_measure=st.number_input("Enter the concavity_worst measure",min_value=0.0,max_value=50.0)
        concave_points_worst_measure=st.number_input("Enter the concave_points_worst measure",min_value=0.0,max_value=10.0)
        symmetry_worst_measure=st.number_input("Enter the symmetry_worst measure",min_value=0.0,max_value=5.0)
        fractal_dimension_worst_measure=st.number_input("Enter the fractal_dimension_worst measure",min_value=0.0,max_value=5.0)
        
        data={'radius_measure':radius_measure,
              'texture_measure':texture_mean,
              'perimeter_measure':perimeter_measure,
              'area_measure':area_measure,
              'smoothness_measure':smoothness_measure,
              'compactness_measure':compactness_measure,
              'concave_measure':concave_measure,
              'concave_point_measure':concave_points_measure,
              'symmetry_measure':symmetry_measure,
              'fractal_diamension_measure':fractal_diamension_measure,
              'radius_seen_measure':radius_seen_measure,
              'texture_seen_measure':texture_seen_measure,
              'perimeter_seen_measure':perimeter_seen_measure,
              'area_seen_measure':area_seen_measure,
              'smoothness_seen_measure':smoothness_seen_measure,
              'compactness_seen_measure':compactness_seen_measure,
              'concavity_seen_measure':concavity_seen_measure,
              'concave_points_seen_measure':concave_points_seen_measure,
              'symmetry_seen_measure':symmetry_seen_measure,
              'fractal_dimension_seen_measure':fractal_dimension_seen_measure,
              'radius_worst_measure':radius_worst_measure,
              'texture_worst_measure':texture_worst_measure,
              'perimeter_worst_measure':perimeter_worst_measure,
              'area_worst_measure':area_worst_measure,
              'smoothness_worst_measure':smoothness_worst_measure,
              'compactness_worst_measure':compactness_worst_measure,
              'concavity_worst_measure':concavity_worst_measure,
              'concave_points_worst_measure':concave_points_worst_measure,
              'symmetry_worst_measure':symmetry_worst_measure,
              'fractal_dimension_worst_measure':fractal_dimension_worst_measure,
              }
        features=pd.DataFrame(data,index=[0])
        return features
    import numpy as np
    import pandas as pd
    df=pd.read_csv('breast-cancer.csv')
    # df.head()

    df=df.drop('id',axis=1)

    def mapping(data,feature):
        featureMap=dict()
        count=0
        for i in sorted(data[feature].unique(),reverse=True):
            featureMap[i]=count
            count=count+1
        data[feature]=data[feature].map(featureMap)
        return data

    df=mapping(df,feature="diagnosis")

    X=df.drop(["diagnosis"],axis=1)
    y=df["diagnosis"]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()

    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    import keras
    from keras import Sequential
    from tensorflow.python.keras.layers import Dense

    model=Sequential()
    model.add(Dense(30,activation='relu'))
    model.add(Dense(15,activation='relu' ))
    model.add(Dense(1,activation='sigmoid' ))
    model.compile(loss='binary_crossentropy',optimizer = 'adam')

    model.fit(x=X_train,y=y_train,epochs=100,validation_data=(X_test,y_test))


    uip=user_input_parameters()
    st.subheader("User Input Parameters")
    st.write(uip)
    prediction=model.predict(uip)
    st.write(prediction)
    if prediction>0.5:
        st.write("The cancer has been clasified as benign type")
    else:
        st.write("The cancer has been clasified as malignent type")
    
    