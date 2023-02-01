"""
Module: face_recognition.py
Author: Amandeep Singh
Date Creted: 31-Jan-2023
Date Modified: 31-Jan-2023

Description:
    module to predict the gender by looking at face image
    - module executes the inference pipeline which consists of
        - import models
        - read image
        - convert to gray scale
        - crop the face using haar cascade classifier
        - normalize the face image
        - create eigen image
        - predict the face using model
        - return the results
"""

import os
import pickle
import argparse
import cv2


def _haar_model():
    """
    load the haar cascade model
    """

    haar = cv2.CascadeClassifier(
        './model/haarcascade_frontalface_default.xml'
    )

    return haar


def _pca_models():
    """
    load the pca models
    """

    f = open(os.path.join(
        os.getcwd(),
        './model/pca_dict.pickle'), 'rb')
    pca_models = pickle.load(
        f
    )
    f.close()

    model_pca = pca_models['pca']  # PCA model
    mean_face_arr = pca_models['mean_face']  # Mean face

    return model_pca, mean_face_arr


def _svm_model():
    """
    load the svm model
    """

    f = open(os.path.join(
        os.getcwd(),
        './model/model_svm.pickle'), 'rb')
    model_svm = pickle.load(
        f
    )
    f.close()

    return model_svm


def _resize_image(img, dimension):
    """
    function will resize the images to the given dimension
        - read image
        - convert BGR to grayscale
        - resize the image to given dimension

    Arguments:
        path: path of the image
        dimension: new resize dimension

    Returns:
        image: resized

    """

    try:
        size = img.shape[0]
        if size >= dimension:
            img_resize = cv2.resize(
                img,
                (dimension, dimension),
                cv2.INTER_AREA
            )
        else:
            img_resize = cv2.resize(
                img,
                (dimension, dimension),
                cv2.INTER_CUBIC
            )

        return img_resize
    except BaseException:
        None


def print_report(model_predictions):
    """print the predictions"""

    for i in range(len(model_predictions)):
        pred_face = model_predictions[i]['prediction']
        score = model_predictions[i]['score']
        text = f"Gender: {pred_face}, Prob: {round(score*100, 1)}%"
        print(text)


def face_recognition_pipeline(img_pth=None, img=None):
    """
    to execute the inference pipeline and get the
    prediction as male or female
    
    Arguments:
        Either the path of image, or image array

        img_pth: Path of the image
        img: image array

    Returns:
        output dictionary with predicition, detected face, probabilties

    """

    # load models
    haar = _haar_model()
    model_svm = _svm_model()
    model_pca, mean_face_arr = _pca_models()

    # read image

    if img is not None:
        img = img
        img_rgb = img.copy()
    else:
        img = cv2.imread(img_pth)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB image
    img_rgb_cp = img_rgb.copy()  # Duplicate for face rect
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)  # GRAY image

    # detect faces
    face_rect = haar.detectMultiScale(img_gray, 1.5, 3)

    predictions = []
    for x, y, w, h in face_rect:

        # get the region of intrest (roi)
        # cv2.rectangle(
        #     img_rgb_cp, 
        #     (x, y), 
        #     (x+w, y+h), 
        #     (0, 255, 0), 
        #     2
        # )
        img_roi = img_gray[y:y + h, x:x + w]
        img_rgb_roi = img_rgb[y:y + h, x:x + w]

        # normalization (0-1)
        img_roi = img_roi / 255

        # resize images (100, 100)
        img_roi = _resize_image(img_roi, 100)

        # Flattening (1 x 10000)
        img_roi = img_roi.reshape(1, 10000)

        # step-07: subtract the mean
        img_roi = img_roi - mean_face_arr

        # get eigen image
        img_eigen = model_pca.transform(img_roi)

        # Eigen Image for Visualization
        img_eigen_vis = (
            model_pca
            .inverse_transform(img_eigen)
            .reshape((100, 100))
        )

        # pass to ml model (svm) and get predictions
        pred_face = model_svm.predict(img_eigen)
        pred_probs = model_svm.predict_proba(img_eigen)
        pred_probs_max = pred_probs.max()

        # image text
        text = f"{pred_face[0]}, {round(pred_probs_max*100, 1)}%"

        # define the colors based on report
        # draw borders on predicted image
        if pred_face == 'male':
            face_color = (0, 255, 255)
        else:
            face_color = (255, 0, 255)
        
        cv2.rectangle(
            img_rgb_cp, 
            (x, y), 
            (x+w, y+h), 
            face_color, 
            1
        )
        # cv2.rectangle(
        #     img_rgb_cp, 
        #     (x, y-30), 
        #     (x+w, y), 
        #     face_color, 
        #     -1
        # )
        # cv2.putText(
        #     img_rgb_cp, 
        #     text, 
        #     (x, y), 
        #     cv2.FONT_HERSHEY_PLAIN, 
        #     1, 
        #     (255, 255, 255), 
        #     1
        # )

        # Results
        output = {
            'pred_image': img_rgb_cp,
            'roi_gray': img_roi,
            'roi_rgb': img_rgb_roi,
            'eig_img': img_eigen,
            'eig_img_vis': img_eigen_vis,
            'prediction': pred_face[0],
            'score': pred_probs_max
        }
        predictions.append(output)

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict the gender from faces in images"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        help="Path of the image from which gender based on faces need to be detected"
    )
    args = parser.parse_args()

    # get the inference
    predicitons = face_recognition_pipeline(args.image_path)
    print_report(predicitons)
