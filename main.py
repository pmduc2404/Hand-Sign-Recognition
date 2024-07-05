from pred_result import *
import cv2
import warnings
warnings.filterwarnings('ignore')

## Capturing the video sequence
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
num_frames = 0
bg = None
text_pred = ""


# cập nhật nền:
def run_avg(img, aweight = 0.5):
    global bg
    if bg is None:
        bg = img.copy().astype('float')
        return cv2.accumulateWeighted(img,bg,aweight)

# Hàm phân đoạn tay trong ảnh và phân ngưỡng:
def segment(img,thres=25):
    global bg
    diff = cv2.absdiff(bg.astype('uint8'),img)
    _, thresholded = cv2.threshold(diff,thres,255,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return
    else:
        segmented = max(contours,key = cv2.contourArea)
    return (thresholded,segmented, contours)

# mở webcam
while(cap.isOpened()):
    ret, frame = cap.read()

    if ret ==True:
        # lật ảnh để có hướng đúng
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        # chọn vùng ảnh để lấy cử chỉ tay
        roi = frame[100:300, 300:500]
        # chuyển sang ảnh xám và làm mờ với GaussianBlur
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        
        if num_frames < 30:
            # cập nhật nền trong 30 frame đầu tiên
            run_avg(gray)
        else:
            # phân đoạn để phát hiện cử chỉ tay
            hand = segment(gray)

            if hand is not None:
                (thresholded, segmented,contours) = hand
                # vẽ đường viền quanh tay
                cv2.drawContours(clone, [segmented + (300, 100)], -1, (0, 0, 255))
                # hiển thị ảnh phân đoạn để theo dõi quá trình
                cv2.imshow("Thesholded", thresholded)
                cv2.imshow("gray", gray)
                contours, _= cv2.findContours(thresholded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

                for cnt in contours:
                    cv2.drawContours(clone, [cnt + (300, 100)], -1, (0, 0, 255))
                    if cv2.contourArea(cnt) > 5000:
                        print("Hand detecting for prediction")
                        if num_frames % 30 == 0:
                            pred = get_prediction(thresholded)
                            text_pred = str(pred)
                    
                        
        cv2.rectangle(clone, (300, 100), (500, 300), (0, 255, 0), 2)
        cv2.putText(clone,text_pred, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        num_frames += 1
        cv2.imshow('frame', clone)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cv2.destroyAllWindows()
