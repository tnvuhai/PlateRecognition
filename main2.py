import math

import numpy as np
import cv2
import imutils
import pytesseract
import pandas as pd
import time
from pathlib import Path
import argparse
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5) #kích cỡ càng to thì càng mờ
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
n = 1

Min_char = 0.01
Max_char = 0.09

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
config = ('-l eng --oem 1 --psm 3')


def GetAgrument():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image', default='./samples/1.jpg')

    return arg.parse_args()


args = GetAgrument()
img_path = Path(args.image_path)


def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    # màu sắc, độ bão hòa, giá trị cường độ sáng
    # Không chọn màu RBG vì vd ảnh màu đỏ sẽ còn lẫn các màu khác nữa nên khó xđ ra "một màu"
    return imgValue


def maximizeContrast(imgGrayscale):
    # Làm cho độ tương phản lớn nhất
    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # tạo bộ lọc kernel

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement,
                                 iterations=10)  # nổi bật chi tiết sáng trong nền tối
    # cv2.imwrite("tophat.jpg",imgTopHat)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement,
                                   iterations=10)  # Nổi bật chi tiết tối trong nền sáng
    # cv2.imwrite("blackhat.jpg",imgBlackHat)
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    # cv2.imshow("imgGrayscalePlusTopHatMinusBlackHat",imgGrayscalePlusTopHatMinusBlackHat)
    # Kết quả cuối là ảnh đã tăng độ tương phản
    return imgGrayscalePlusTopHatMinusBlackHat


# đường dẫn đến file tesseract.exe trong thư mục cài đặt của phần mềm
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# đọc ảnh bằng opencv2
image = cv2.imread(str(img_path))
# resize ảnh xuống 500 px cho thuận tiện xử lý với imutils.resize()
image = imutils.resize(image, width=500)
# hiển thị hình ảnh bằng imshow của opencv2
cv2.imshow("Anh goc", image)
cv2.waitKey(0)

# chuyển đổi sang ảnh đa cấp xám
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imgGrayscaleplate = extractValue(image)
imgMaxContrastGrayscale = maximizeContrast(imgGrayscaleplate) #để làm nổi bật biển số hơn, dễ tách khỏi nền
height, width = imgGrayscaleplate.shape

imgBlurred = np.zeros((height, width, 1), np.uint8)
imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, (5, 5), 0)
# cv2.imwrite("gauss.jpg",imgBlurred)
# Làm mịn ảnh bằng bộ lọc Gauss 5x5, sigma = 0

imgThreshplate = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                  19, 9)
# hiển thị ảnh
cv2.imshow("Anh da cap xam", imgGrayscaleplate)

# giảm nhiễu của ảnh đa cấp xám bằng opencv2
'''
Giảm nhiễu trong ảnh xám, đồng thời làm mịn ảnh hơn bằng BilateralFilter (làm mờ song phương).
BilateralFilter là phương pháp làm mịn ảnh của OpenCv2, làm mịn ảnh mà vẫn giữ lại được biên của ảnh.
BilateralFilter sử dụng một bộ lọc Gauss với khoảng cách đến điểm trung tâm, đảm bảo chỉ có các điểm ở gần tham gia vào 
giá trị của điểm ảnh trung tâm. Tuy vậy nó sử dụng thêm một hàm Gauss cho mức xám, đảm bảo chỉ các điểm ảnh có mức xám 
tương đồng với điểm ảnh trung tâm tham gia vào quá trình làm mịn. Vì thế bộ lọc Bilateral bảo toàn được các đường biên 
trong ảnh bởi vì điểm ảnh ở biên có sự thay đổi về mức xám rất rõ ràng. Hơn nữa, thay vì hoạt động trên các kênh màu một
 cách riêng rẽ như bộ lọc trung bình hay bộ lọc Gauss, bộ lọc Bilateral có thể thi hành việc đo đạc màu sắc có chủ đích 
 trong không gian màu CIE-Lab, làm mượt màu và bảo toàn các biên theo hướng phù hợp hơn với nhận thức con người.

Tuy vậy, bộ lọc Bilateral cũng có nhược điểm là chậm hơn các bộ lọc khác.
http://people.csail.mit.edu/sparis/bf_course/
'''
gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
cv2.imshow("Anh lam min BF", gray_image)
cv2.waitKey(0)

# phát hiện biên của ảnh đã làm mịn bằng Canny
edged = cv2.Canny(gray_image, 170, 200)
cv2.imshow("Bien anh", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
###### vẽ contour và lọc biển số  #############
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Lấy 10 contours có diện tích lớn nhất

# cv2.drawContours(img, contours, -1, (255, 0, 255), 3) # Vẽ tất cả các ctour trong hình lớn

screenCnt = []
for c in contours:
    peri = cv2.arcLength(c, True)  # Tính chu vi
    approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
    [x, y, w, h] = cv2.boundingRect(approx.copy())
    ratio = w / h
    # cv2.putText(img, str(len(approx.copy())), (x,y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
    # cv2.putText(img, str(ratio), (x,y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
    if (len(approx) == 4):
        screenCnt.append(approx)

        cv2.putText(image, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

if screenCnt is None:
    detected = 0
    print("No plate detected")
else:
    detected = 1

if detected == 1:

    for screenCnt in screenCnt:
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe


        ####################################

        ########## Cắt biển số ra khỏi ảnh và xoay ảnh ################

        mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
        # cv2.imshow("new_image",new_image)
        # Now crop
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))

        roi = image[topx:bottomx, topy:bottomy]
        imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
        ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2




        roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
        imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

        ####################################

        #################### Tiền xử lý ảnh đề phân đoạn kí tự ####################
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
        cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow(str(n + 20), thre_mor)
        cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)  # Vẽ contour các kí tự trong biển số
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ##################### Lọc vùng kí tự #################
        char_x_ind = {}
        char_x = []
        height, width, _ = roi.shape
        roiarea = height * width

        for ind, cnt in enumerate(cont):
            (x, y, w, h) = cv2.boundingRect(cont[ind])
            ratiochar = w / h
            char_area = w * h
            # cv2.putText(roi, str(char_area), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
            # cv2.putText(roi, str(ratiochar), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)

            if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                    x = x + 1
                char_x.append(x)
                char_x_ind[x] = ind

                # cv2.putText(roi, str(char_area), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)

        ############ Cắt và nhận diện kí tự ##########################

        char_x = sorted(char_x)
        strFinalString = ""
        first_line = ""
        second_line = ""

        for i in char_x:
            (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

            imgROI = thre_mor[y:y + h, x:x + w]  # cắt kí tự ra khỏi hình

            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # resize lại hình ảnh
            npaROIResized = imgROIResized.reshape(
                (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # đưa hình ảnh về mảng 1 chiều

            # if text:
            #     break
            # # cHUYỂN ảnh thành ma trận có 1 hàng và số cột là tổng số điểm ảnh trong đó
            # npaROIResized = np.float32(npaROIResized)  # chuyển mảng về dạng float
            # # _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
            # #                                                         k=3)  # call KNN function find_nearest; neigh_resp là hàng xóm
            # # strCurrentChar = str(chr(int(npaResults[0][0])))  # Lấy mã ASCII của kí tự đang xét
            # # cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)
            # #
            # # if (y < height / 3):  # Biển số 1 hay 2 hàng
            # #     first_line = first_line + strCurrentChar
            # # else:
            # #     second_line = second_line + strCurrentChar

        roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
        cv2.imshow(str(n), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        text = pytesseract.image_to_string(roi, config=config)
        print("\n License Plate " + str(n) + " is: " + text + " - " + second_line + "\n")

        # cv2.putText(img, first_line + "-" + second_line ,(topy ,topx),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
        n = n + 1


