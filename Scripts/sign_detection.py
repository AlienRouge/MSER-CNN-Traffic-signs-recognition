import os
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import flake8


def contrast_stabilization(_image):
    lab = cv2.cvtColor(_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def bootstrap(windows):
    if len(windows) == 1:
        return windows

    epsilon = 10
    # 35
    groups = {}
    indexes = list(range(len(windows)))
    i = 0
    while i < len(windows):
        groups[i] = [i]
        indexes.remove(i)
        to_remove = []
        for j in indexes:
            if (np.abs(np.array(windows[i]) - np.array(windows[j])) < epsilon).all():
                groups[i].append(j)
                to_remove.append(j)
        for k in to_remove:
            indexes.remove(k)
        if indexes:
            i = indexes[0]
        else:
            break

    res_windows = []
    for key in groups:
        xmin = np.min([windows[ind][0] for ind in groups[key]])
        ymin = np.min([windows[ind][1] for ind in groups[key]])
        xmax = np.max([windows[ind][2] for ind in groups[key]])
        ymax = np.max([windows[ind][3] for ind in groups[key]])
        res_windows.append((xmin, ymin, xmax, ymax))

    return res_windows


def get_valid_mser_windows(_image):
    img = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create(_delta=2)
    mser_areas, _ = mser.detectRegions(img)
    padding = 8

    mser_windows = list()
    for area in mser_areas:
        df_area = pd.DataFrame(area)
        xmin, ymin = df_area.min()
        xmax, ymax = df_area.max()

        xmin -= padding
        ymin -= padding
        xmax += padding
        ymax += padding

        if xmin >= 0 and ymin >= 0 and xmax <= img.shape[1] and ymax <= img.shape[0]:
            mser_windows.append([xmin, ymin, xmax, ymax])

    validated_areas = list()
    min_area_size = 35
    max_area_size = 200
    max_proportion = 1.2

    # ФИЛЬТРАЦИЯ КООРДИНАТ (xmin ymin xmax ymax)
    for win in mser_windows:
        size_x = float(win[2] - win[0])
        size_y = float(win[3] - win[1])
        if (size_x < min_area_size) or (size_y < min_area_size) or (size_x > max_area_size) or (
                size_y > max_area_size) or \
                (size_x / size_y > max_proportion) or (size_y / size_x > max_proportion):
            continue
        win[0] += padding
        win[1] += padding
        win[2] -= padding
        win[3] -= padding

        # Отступ 15%
        paddingX = round(size_x * 0.1)
        paddingY = round(size_y * 0.1)
        win[0] -= paddingX
        win[1] -= paddingY
        win[2] += paddingX
        win[3] += paddingY
        validated_areas.append(win)

    # ПОДСВЕТКА КОНТУРОВ
    img_test = _image.copy()
    for area in validated_areas:
        cv2.rectangle(img_test, (area[0], area[1]), (area[2], area[3]), (100, 255, 0), 1)
    cv2.imshow("MSER areas", img_test)
    print(len(validated_areas))

    # ПОДГОТОВКА ИТОГОВЫХ ИЗОБРАЖЕНИЙ
    window_tuples = []  # КООРДИНАТЫ
    window_areas = []  # ОБЛАСТИ

    for window in validated_areas:
        window_tuples.append(window)
        window_areas.append(_image[window[1]:window[3], window[0]:window[2]])

    window_areas = [img for img in window_areas if img.shape[0] > 0 and img.shape[1] > 0]
    return window_tuples, window_areas


def detect_traffic_signs(input_image, detect_model, IMAGE_SIZE):
    DETECT_CLASSES = ['Priority road', 'Yield', 'Round', 'Square', "DontEnter", 'Trash', 'Triangle']
    TRASH_INDEX = DETECT_CLASSES.index("Trash")

    # ЛОКАЛИЗАЦИЯ ОБЛАСТЕЙ И ПРИВЕДЕНИЕ К ЗАДАННОМУ РАЗМЕРУ
    image = contrast_stabilization(input_image)
    test_image1 = image.copy()
    test_image2 = image.copy()
    tuples, areas = get_valid_mser_windows(image)
    resized_images = np.array([cv2.resize(image, IMAGE_SIZE) for image in areas])

    # СОХРАНЕНИЕ ПОЛУЧЕННЫХ ОБЛАСТЕЙ В ПАПКУ
    mser_path = 'G:/PyProjects/OldNet/Data/MSER/'
    filelist = [f for f in os.listdir(mser_path)]
    for f in filelist:
        os.remove(os.path.join(mser_path, f))
    for image in range(len(resized_images)):
        cv2.imwrite(mser_path + str(tuples[image]) + ".png", areas[image])
    print("MSER AREAS:", resized_images.shape)

    # ЗАГРУЗКА МОДЕЛИ
    model = tf.keras.models.load_model(detect_model)
    model_img = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in resized_images])
    model_img = model_img / 255

    filelist = [f for f in os.listdir("G:/PyProjects/OldNet/Data/Output_mser/")]
    for f in filelist:
        os.remove(os.path.join("G:/PyProjects/OldNet/Data/Output_mser/", f))

    regions = []
    indexes = []
    images = []
    for x in range(len(model_img)):
        img = np.expand_dims(model_img[x], axis=0)
        predict = model.predict(img)

        prob = np.max(predict)
        if prob > .5:
            index = np.nanargmax(predict)
            line = DETECT_CLASSES[int(index)] + " " + str(round(prob, 2) * 100) + "%"
            if index != TRASH_INDEX:
                print(line)
                cv2.putText(test_image1, line,
                            org=(tuples[x][0], tuples[x][1] - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.3,
                            color=(0, 0, 255),
                            thickness=1)

                regions.append(tuples[x])
                indexes.append(index)
                images.append(areas[x])

                cv2.rectangle(test_image1, (tuples[x][0], tuples[x][1]), (tuples[x][2], tuples[x][3]),
                              (0, 255, 255), 1)
                cv2.imwrite("G:/PyProjects/OldNet/Data/Output_mser/" + str(tuples[x]) + ".png", areas[x])
    cv2.imshow("D-CNN", test_image1)

    output_regions = bootstrap(regions)

    numbers = []
    for i in regions:
        if regions.index(i) in numbers:
            continue
        for j in output_regions:
            if i[0] == j[0] and i[1] == j[1] and i[2] == j[2] and i[3] == j[3]:
                numbers.append(regions.index(i))
                break

    output_indexes = [indexes[index] for index in numbers]
    output_images = [images[index] for index in numbers]

    for reg in range(len(output_images)):
        line = DETECT_CLASSES[output_indexes[reg]]
        cv2.rectangle(test_image2, (output_regions[reg][0], output_regions[reg][1]),
                      (output_regions[reg][2], output_regions[reg][3]),
                      (0, 200, 255), 1)
        cv2.putText(test_image2, line,
                    org=(output_regions[reg][0], output_regions[reg][1]-10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.3,
                    color=(0, 255, 0),
                    thickness=1)
        cv2.imwrite("G:/PyProjects/OldNet/Data/OUTPUT/" + str(DETECT_CLASSES[output_indexes[reg]]) + "_" + str(
            output_regions[reg][0]) + ".png", output_images[reg])
    cv2.imshow("BS", test_image2)
    cv2.waitKey(0)

    return output_regions, output_images
