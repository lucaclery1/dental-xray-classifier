# dental-xray-classifier
This is a machine learning project that classifies periapical dental xrays into:
Healthy, Crowned or Root canal.
This project compares a logistic regression model with a convolutional neural network (CNN) model. 
The CNN model is deployed through an interactive website and is able to take a periapical xray of a tooth as input and returns softmax percentages for each class. 
## Website example screenshots and live demo video:
<img width="1512" height="982" alt="Screenshot 2026-08-30 at 12 46 10" src="https://github.com/user-attachments/assets/b38518cf-684e-4ee0-8c67-e235a423b54c" />
<img width="1512" height="982" alt="Screenshot 2026-08-30 at 12 45 00" src="https://github.com/user-attachments/assets/93eddb15-211d-4f2e-8e43-c37e88111cd4" />



https://github.com/user-attachments/assets/5563a6d1-6a6b-4b52-8441-d2741c8b506c


project aims:
- build a machine learning model using PCA and logistic regression
- build a CNN to learn features from xrays
- compare and evaluate both models through training, validation and test sets
- use accuracy, balanced accuracy and macro F1 scores to evaluate
- deploy best model (CNN) through a publicly available website.

## Dataset Information:
This project uses the [DentIRO dataset](https://doi.org/10.6084/m9.figshare.32086377).
Dataset contains 5300 labelled intraoral radiograph images
all images with the label "caries" were excluded from the project as initial tests showed that they could not reliably be identified as they contain minor differences to healthy teeth.


total images used: 4696

dataset class split:
- healthy: 1486 images
- crowned: 779 images
- root canal: 2431 images

dataset test/validation/test split:
- train: 3296 images
- validation: 697 images
- test: 703 images

dataset was split by `patient_id` so images from the same patient are not in different dataset partitions

Preprocessing process:
- convert to greyscale
- crop
- resize
- normalise
- change colour channel to greyscale

To make logistic regression model:
- resize images to 32x32
- flatten to 1D vector
- use PCA
- use validation macro F1 score to select 30 PCA components

To make CNN model:
- Adam optimisation used
- accounted for imbalance in amount of images in each class by using class weights
- early stopping
- batches of 32 images

## Results

CNN clearly outperforms the logistic regression model

Logistic regression results:
- Test accuracy: 71.0%
- Balanced accuracy: 71.6%
- Macro F1 score:70.2%

CNN results:
- Test accuracy:96.2%
- Balanced accuracy: 95.6%
- Macro F1 score: 95.5%

CNN model incorrectly classified 27 of the 701 test images

## website instructions:

- open [DentIRO dataset](https://doi.org/10.6084/m9.figshare.32086377)  in web browser
- Upload PNG, JPG or JPEG of dental xray
- press classify image button to classify image using CNN model
- view predicted class and observe softmax scores

## Technologies Used :
- Python
- TensorFlow
- Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Pillow
- Streamlit
- Jupyter Notebook
- Git/Github
