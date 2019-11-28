import cv2
import numpy as np
import matplotlib.pylab as plt
import glob

# 동영상 파일 경로
video_path = './video1.mp4'

cap = cv2.VideoCapture(video_path)
video_file = video_path.split('/')[-1]
video_name = video_file.split('.')[0]

if not cap.isOpened() :
    print("can't open video.")

else : 
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000/fps)

    index = 1
    while True :
        ret, img = cap.read()   

        if ret :
            img2 = img.copy()
            img3 = img.copy()
            h, w = img.shape[:2]

            # HSV 영상으로 변환
            img2 = cv2.bilateralFilter(img2, 9, 105, 105)
            r, g, b = cv2.split(img2)
            equalize1= cv2.equalizeHist(r)
            equalize2= cv2.equalizeHist(g)
            equalize3= cv2.equalizeHist(b)
            equalize = cv2.merge((equalize1, equalize2, equalize3))
            img2 = equalize
            hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

            # 색상별 영역 지정
            red1 = np.array([165, 50, 50])
            red2 = np.array([180, 255, 255])

            # 색상에 따른 마스크 생성
            mask_red = cv2.inRange(hsv, red1, red2)

            numOfLabels, img_label, stats, centroids = cv2.connectedComponentsWithStats(mask_red)

            for idx, centroid in enumerate(centroids) :
                if stats[idx][0] == 0 and stats[idx][1] == 0 :
                    continue

                if np.any(np.isnan(centroid)) :
                    continue

                x, y, w, h, area = stats[idx]
                centerX, centerY = int(centroid[0]), int(centroid[1])

                if area > 80 and abs(w-h) < 5 : # if area > 120 and abs(w-h) < 5 :

                    # 원 검출
                    detected_img = img2[y:y+h, x:x+w]
                    gray_detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2GRAY)                 
                    # circles = cv2.HoughCircles(gray_detected_img, cv2.HOUGH_GRADIENT, 1, 100, param1=200, param2=50, minRadius=10, maxRadius=0)
                    circles = cv2.HoughCircles(gray_detected_img, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=45, minRadius=10, maxRadius=20)
                    
                    if circles is not None :
                        circles = np.uint16(np.around(circles))

                        for i in circles[0, :] :
                            cv2.rectangle(img, (x-8, y-8), (x+w+8, y+h+8), (0, 0, 255), 2)
                            cv2.imwrite('images7/{}_{}.jpg'.format(video_name, index), img3[y-8:y+h+8, x-8:x+w+8])
                            index += 1

            cv2.imshow(video_file, img)
            if cv2.waitKey(1) == 27 :
                cap.release()
                cv2.destroyAllWindows()
        else :
            break

cap.release()
cv2.destroyAllWindows()