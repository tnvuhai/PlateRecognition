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
# hiển thị ảnh
cv2.imshow("Anh da cap xam", gray_image)

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
Biensoxe = None
count = 0
try:
# chạy vòng lập trong các contours đã lọc bên trên
    for c in cnts:
        # tính chu vi của những contours trong lặp, biến True để xác nhận là contours kín
        perimeter = cv2.arcLength(c, True)
        # thuật toán nhằm đơn giản hoá contours, tham số đưa vào là contour, epsilon(độ chính xác),  tham số True để xác
        # định là contours kín
        approx = cv2.approxPolyDP(c, 0.06 * perimeter, True)
        # nếu contours có 4 đỉnh thì lưu vào biến đã tạo ở trên
        if len(approx) == 4:
            Biensoxe = approx
            break
            # lập tức thoát khỏi vòng lập
except Exception as e:
    print(f"Lỗi xảy ra do tìm contours, lỗi cụ thể:{e}")

# bỏ những phần không phải biển số xe bằng cách thay bằng phần tử 0
# dùng hàm np.zeros để trả về bit 0 (tức điểm đen) của đúng hình dạng ảnh
mask = np.zeros(gray_image.shape, np.uint8)
# sau đó vẽ contour đã lọc lên bằng drawContours trên ảnh đen vừa tạo
new_image = cv2.drawContours(mask, [Biensoxe], 0, 255, -1)
cv2.imshow("abc", new_image)
# dùng hàm bitwise_and của cv2 để áp dụng mask điểm đen
new_image = cv2.bitwise_and(image, image, mask=mask)
# sau đó hiển thị cửa sổ và hiển thị ảnh
cv2.namedWindow("Final Image", cv2.WINDOW_NORMAL)
cv2.imshow("Final Image", new_image)

# thiết lập cho tesseract
config = ('-l eng --oem 1 --psm 3')

#dịch chữ từ ảnh ra bằng tesseract
try:
    text = pytesseract.image_to_string(new_image, config=config)
except Exception as e:
    print(f"Không thể convert ra text, lỗi cụ thể: {e}")
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
