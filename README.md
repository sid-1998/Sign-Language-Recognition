# Sign-Language-Recognition

The CNN is trained on a custom dataset containing alphabets A-Y (excluding J) of American Sign Language.

## Usage 

### To run the pretrained model

Run:

```
python Gesture_recognize_sign.py
```

This will start the webcam.Press C then place your hand inside the green box while performing a gesture
and you will get the letter to which the respective gesture corresponds. Press Q to quit.

### To train your own model

Set up the path in the Image_capturing.py file

Run:

```
python Image_capturing.py
```

Place your hand in the green box and press C to start capturing the data.

Now set up the paths in Image_preprocessing.py file to preprocess the dataset.

Then Run:

```
python Image_preprocessing.py
```

After preprocessing set up the path in model.py file to get the preprocessed data for training.

Then Run:

```
python model.py
```

This will create model-<val_acc>.h5 files. Choose the appropriate file for Gesture_recognize_sign.py

Then Run:

```
python Gesture_recognize_sign.py
```

## Demo

![Demo]( ./Demo.gif )
