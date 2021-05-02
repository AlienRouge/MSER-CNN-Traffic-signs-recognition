import tensorflow as tf
import numpy as np
import cv2


def classify_areas(image, input_regions, input_images, classify_model, IMAGE_SIZE):
    CLASSIFICATION_CLASSES = ['1_11_1', '1_11_2', '1_17', '1_23', '1_25', '2_1', '2_4', '3_1', '3_20', '3_24_n40',
                              '3_24_n60', '3_24_n80', '5_19_1', '5_19_2' '5_20', '6_4']

    model = tf.keras.models.load_model(classify_model)

    inputImagesResized = np.array([cv2.resize(image, IMAGE_SIZE) for image in input_images])
    model_img = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in inputImagesResized])
    model_img = model_img / 255

    for x in range(len(model_img)):
        img = np.expand_dims(model_img[x], axis=0)
        predict = model.predict(img)

        prob = np.max(predict)
        if prob > .8:
            index = np.nanargmax(predict)
            line = CLASSIFICATION_CLASSES[int(index)] + " " + str(round(prob, 2) * 100) + "%"
            print(line)
            cv2.putText(image, line,
                        org=(input_regions[x][0], input_regions[x][1] - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 255, 0),
                        thickness=2)

            cv2.rectangle(image, (input_regions[x][0], input_regions[x][1]),
                          (input_regions[x][2], input_regions[x][3]),
                          (0, 100, 255), 1)

            cv2.imwrite("G:/PyProjects/OldNet/Data/RESULTS/" + str(CLASSIFICATION_CLASSES[int(index)]) + str(
                input_regions[x][0]) + ".png", input_images[x])
    cv2.imshow("Output", image)

    cv2.waitKey(0)
