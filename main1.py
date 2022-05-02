import math

import numpy as np
import cv2
import imutils
import sys
import pytesseract
import pandas as pd
import time
from pathlib import Path
import argparse
def GetAgrument():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image', default='./samples/1.jpg')

    return arg.parse_args()

config = ('-l eng --oem 1 --psm 3')
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

n = 1

args = GetAgrument()
img_path = Path(args.image_path)

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
'''Giải thuật phát hiện cạnh Canny - Canny Edge Detection
Trong hình ảnh, thường tồn tại các thành phần như: vùng trơn, góc / cạnh và nhiễu. Cạnh trong ảnh mang đặc trưng quan 
trọng, thường là thuộc đối tượng trong ảnh (object). Do đó, để phát hiện cạnh trong ảnh, giải thuật Canny là một trong 
những giải thuật được dùng nhiều trong Xử lý ảnh.

Giải thuật phát hiện cạnh Canny gồm 4 bước chính sau:

- Giảm nhiễu: Làm mờ ảnh, giảm nhiễu dùng bộ lọc Gaussian kích thước 5x5. Kích thước 5x5 thường hoạt động tốt cho giải 
thuật Canny. Dĩ nhiên bạn cũng có thể thay đổi kích thước của bộ lọc làm mờ cho phù hợp. 
- Tính Gradient và hướng gradient: ta dùng bộ lọc Sobel X và Sobel Y (3x3) để tính được ảnh đạo hàm Gx và Gy. 


- NMS (Non-maximum Suppression): loại bỏ các pixel ở vị trí không phải cực đại toàn cục. Ở bước này, ta dùng một filter 
3x3 lần lượt chạy qua các pixel trên ảnh gradient. Trong quá trình lọc, ta xem xét xem độ lớn gradient của pixel trung 
tâm có phải là cực đại (lớn nhất trong cục bộ - local maximum) so với các gradient ở các pixel xung quanh. 
Nếu là cực đại, ta sẽ ghi nhận sẽ giữ pixel đó lại. Còn nếu pixel tại đó không phải là cực đại lân cận, ta sẽ set độ lớn 
gradient của nó về zero. Ta chỉ so sánh pixel trung tâm với 2 pixel lân cận theo hướng gradient. Ví dụ: nếu hướng 
gradient đang là 0 độ, ta sẽ so pixel trung tâm với pixel liền trái và liền phải nó. Trường hợp khác nếu hướng gradient
là 45 độ, ta sẽ so sánh với 2 pixel hàng xóm là góc trên bên phải và góc dưới bên trái của pixel trung tâm. Tương tự 
cho 2 trường hợp hướng gradient còn lại. Kết thúc bước này ta được một mặt nạ nhị phân

- Lọc ngưỡng: ta sẽ xét các pixel dương trên mặt nạ nhị phân kết quả của bước trước. Nếu giá trị gradient vượt ngưỡng 
max_val thì pixel đó chắc chắn là cạnh. Các pixel có độ lớn gradient nhỏ hơn ngưỡng min_val sẽ bị loại bỏ. Còn các pixel
nằm trong khoảng 2 ngưỡng trên sẽ được xem xét rằng nó có nằm liên kề với những pixel được cho là "chắc chắn là cạnh"
hay không. Nếu liền kề thì ta giữ, còn không liền kề bất cứ pixel cạnh nào thì ta loại. Sau bước này ta có thể áp dụng 
thêm bước hậu xử lý loại bỏ nhiễu (tức những pixel cạnh rời rạc hay cạnh ngắn) nếu muốn.'''

# Xác định các contours từ biên ảnh
# lấy tất cả các contours , RETR_LIST dùng để lấy tất cả các contours nhưng không tạo ra mối quan hệ cha con,
# CHAIN_APPROX_SIMPLE: Biến nhằm xoá các điểm dư thừa trong những contour thu được.
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


'''
Thuật toán Contour
-	Contour được hiểu đơn giản là một đường cong liên kết toàn bộ các điểm liên tục (dọc theo đường biên) mà có cùng màu
 sắc hoặc giá trị cường độ. Nói cách khác Các bạn có thể hiểu contour là “tập các điểm-liên-tục tạo thành một đường cong 
 (curve) (boundary), và không có khoảng hở trong đường cong đó, đặc điểm chung trong một contour là các các điểm có cùng
  /gần xấu xỉ một giá trị màu, hoặc cùng mật độ. Contour là một công cụ hữu ích được dùng để phân tích hình dạng đối 
  tượng, phát hiện đối tượng và nhận dạng đối tượng”.
-	Contour là thuật toán được sử dụng trong xử lý ảnh nhằm tách, trích xuất các đối tượng, tạo điều kiện để các xử lý 
sau được chính xác. Contour rất hữu ích trong phân tích hình dạng, phát hiện vật thể và nhận diện vật thể. 
'''
# Phân loại những contours đã xác định
# lọc lấy những contour có diện tích nhỏ hơn 30

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
# đặt  biến để lưu biển số xe
screenCnt = []
count = 0
# chạy vòng lập trong các contours đã lọc bên trên
for c in cnts:
    # tính chu vi của những contours trong lặp, biến True để xác nhận là contours kín
    perimeter = cv2.arcLength(c, True)
    # thuật toán nhằm đơn giản hoá contours, tham số đưa vào là contour, epsilon(độ chính xác),  tham số True để xác
    # định là contours kín
    approx = cv2.approxPolyDP(c, 0.06 * perimeter, True)  # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
    [x, y, w, h] = cv2.boundingRect(approx.copy())
    ratio = w / h
    # nếu contours có 4 đỉnh thì lưu vào biến đã tạo ở trên
    if len(approx) == 4:
        screenCnt.append(approx)
        cv2.putText(image, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

        # lập tức thoát khỏi vòng lập
if screenCnt is None:
    detected = 0
    print("No plate detected")
else:
    detected = 1

if detected == 1:

    for screenCnt in screenCnt:
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe

        ############## Tìm góc xoay ảnh #####################
        (x1, y1) = screenCnt[0, 0]
        (x2, y2) = screenCnt[1, 0]
        (x3, y3) = screenCnt[2, 0]
        (x4, y4) = screenCnt[3, 0]
        array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        sorted_array = array.sort(reverse=True, key=lambda x: x[1])
        (x1, y1) = array[0]
        (x2, y2) = array[1]
        doi = abs(y1 - y2)
        ke = abs(x1 - x2)
        angle = math.atan(doi / ke) * (180.0 / math.pi)

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

        if x1 < x2:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
        else:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

        roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
        imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
        roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
        imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

        ####################################

        #################### Tiền xử lý ảnh đề phân đoạn kí tự ####################
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
        cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow(str(n + 20), thre_mor)
        cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)  # Vẽ contour các kí tự trong biển số
        try:
            text = pytesseract.image_to_string(thre_mor, config=config)
        except Exception as e:
            print(f"Không thể convert ra text, lỗi cụ thể: {e}")
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

            if (0.01 * roiarea < char_area < 0.09 * roiarea) and (0.25 < ratiochar < 0.7):
                if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                    x = x + 1
                char_x.append(x)
                char_x_ind[x] = ind

                # cv2.putText(roi, str(char_area), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)

        ############ Cắt và nhận diện kí tự ##########################

        # char_x = sorted(char_x)
        # strFinalString = ""
        # first_line = ""
        # second_line = ""
        #
        # for i in char_x:
        #     (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
        #     cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #
        #     imgROI = thre_mor[y:y + h, x:x + w]  # cắt kí tự ra khỏi hình
        #
        #     imgROIResized = cv2.resize(imgROI, (20, 30))  # resize lại hình ảnh
        #     npaROIResized = imgROIResized.reshape(
        #         (1, 20 * 30))  # đưa hình ảnh về mảng 1 chiều
        #     # cHUYỂN ảnh thành ma trận có 1 hàng và số cột là tổng số điểm ảnh trong đó
        #     npaROIResized = np.float32(npaROIResized)  # chuyển mảng về dạng float
        #     _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
        #                                                             k=3)  # call KNN function find_nearest; neigh_resp là hàng xóm
        #     strCurrentChar = str(chr(int(npaResults[0][0])))  # Lấy mã ASCII của kí tự đang xét
        #     cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)
        #
        #     if (y < height / 3):  # Biển số 1 hay 2 hàng
        #         first_line = first_line + strCurrentChar
        #     else:
        #         second_line = second_line + strCurrentChar
        #
        # print("\n License Plate " + str(n) + " is: " + first_line + " - " + second_line + "\n")
        # roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
        # cv2.imshow(str(n), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        # cv2.putText(img, first_line + "-" + second_line ,(topy ,topx),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
        n = n + 1

img = cv2.resize(image, None, fx=0.5, fy=0.5)
cv2.imshow('License plate', img)

cv2.waitKey(0)


# # bỏ những phần không phải biển số xe bằng cách thay bằng phần tử 0
# # dùng hàm np.zeros để trả về bit 0 (tức điểm đen) của đúng hình dạng ảnh
# mask = np.zeros(gray_image.shape, np.uint8)
# # sau đó vẽ contour đã lọc lên bằng drawContours trên ảnh đen vừa tạo
# new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
# cv2.imshow("abc", new_image)
# # dùng hàm bitwise_and của cv2 để áp dụng mask điểm đen
# new_image = cv2.bitwise_and(image, image, mask=mask)
# # sau đó hiển thị cửa sổ và hiển thị ảnh
# cv2.namedWindow("Final Image", cv2.WINDOW_NORMAL)
# cv2.imshow("Final Image", new_image)

# thiết lập cho tesseract


#dịch chữ từ ảnh ra bằng tesseract

# lưu data từ ảnh vào csv
try:
    raw_data = {'date': [time.asctime(time.localtime(time.time()))], '': [text]}
    df = pd.DataFrame(raw_data)
    df.to_csv('data.csv', mode='a')
except Exception as e:
    print(f"Không thể tạo file csv, lỗi cụ thể: {e}")

# in ra biển số xe
print(text)

cv2.waitKey(0)
cv2.destroyAllWindows()
