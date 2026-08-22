import cv2

# 嘗試打開預設攝影機
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("錯誤：完全無法連接到攝影機！")
else:
    print("成功連接攝影機，嘗試讀取畫面...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("警告：已連接攝影機，但無法獲取畫面內容！")
        break
        
    cv2.imshow("Camera Test", frame)
    
    # 按下 ESC 鍵退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()