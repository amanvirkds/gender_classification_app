import os
import cv2
from flask import render_template, request
from src.face_recognition import face_recognition_pipeline, print_report
import matplotlib.image as matimg

UPLOAD_FOLDER = 'static/upload'

def index():
    return render_template('index.html')


def app():
    return render_template('app.html')

def genderapp():

    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename
        # save image to upload folder
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)

        # get predictions
        model_predictions = face_recognition_pipeline(path)
        pred_image = model_predictions[0]['pred_image'] # predicted image
        pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
        #pred_image_name = f'{filename[:3]}_pred_image.jpg'
        pred_image_name = "prediction_image.jpg"
        cv2.imwrite(f'./static/predict/{pred_image_name}', pred_image)

        model_results = dict()
        model_results['pred_image'] = pred_image_name
        report = []
        for i in range(len(model_predictions)):
            obj_rgb = model_predictions[i]['roi_rgb'] # rgb scale
            obj_eig = model_predictions[i]['eig_img_vis'] # eigen image
            pred_face = model_predictions[i]['prediction']
            score = model_predictions[i]['score']
            #cv2.imwrite(f'./static/predict/{filename[:3]}_face_{i}.jpg', obj_rgb)

            # save rgb and eigen image
            rgb_img_name = f"{filename[:3]}_face_{i}.jpg"
            matimg.imsave(f'./static/predict/{rgb_img_name}', obj_rgb)

            eigen_img_name = f"{filename[:3]}_face_eigen_{i}.jpg"
            matimg.imsave(f'./static/predict/{eigen_img_name}', obj_eig)
            text = f"{pred_face}, {round(score*100, 1)}%"

            # save report
            report.append([
                rgb_img_name,
                eigen_img_name,
                pred_face,
                str(round(score*100, 1)) + "%"
            ])
        model_results['report'] = report

        return render_template(
            'gender.html', 
            fileupload=True,
            report=model_results['report']) # POST Request

    return render_template(
        'gender.html', 
        fileupload=False) # GET Request

