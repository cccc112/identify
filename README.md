##✋ Hand Gesture Drawing Recognition System
##✍️ 空中手勢繪圖辨識系統

#⚠️ 注意 / Note:
本系統對光線變化較為敏感，請在光線均勻的環境中操作，以確保辨識效果穩定。
This system is sensitive to lighting conditions. Please operate in a well-lit and evenly illuminated environment to ensure stable recognition.

#🧠 功能簡介 / Features
#✍️ 手勢空中繪圖 / Air drawing using hand gestures

#🔢 支援數字辨識（0–9） / Digit recognition (0–9)

#🔤 支援英文字母辨識（A–Z） / Alphabet recognition (A–Z)

#📷 攝影機即時影像處理 / Real-time webcam image processing

#🤚 利用 MediaPipe 偵測與追蹤手部 / Hand tracking with MediaPipe

#🧽 擦除、清除畫布功能 / Eraser and canvas reset

#🧠 使用 TensorFlow Keras 預訓練模型進行辨識 / Digit & letter prediction via pre-trained TensorFlow models

#📁 專案結構 / Project Structure

├── digit.py               # 數字/符號手勢繪圖主程式 / Main script for digit/symbol recognition
├── letter.py              # 英文字母手勢繪圖主程式 / Main script for letter recognition
├── best_model.h5          # 手寫數字模型（請自行放置） / Digit model (place manually)
├── symbol.h5              # 符號辨識模型（選用） / Symbol model (optional)
├── augmented_model.h5     # 字母辨識模型（請自行放置） / Alphabet model (place manually)
操作說明 / How to Use
執行 digit.py 或 letter.py / Run either digit.py or letter.py

用食指在空中進行繪圖 / Draw in the air using your index finger

食指與中指靠近 → 停止繪圖 / Pinch index and middle finger to stop drawing

握拳 → 啟動橡皮擦功能 / Fist gesture to erase

點擊畫面上方按鈕清除畫布 / Click Clear-c to clear canvas

可切換數字/符號或字母模式 / Switch between digit/symbol or letter mode

#💡 建議環境 / Environment Tips
#✅ 光線充足且均勻 / Ensure even and sufficient lighting

#❌ 避免強背光或手部陰影 / Avoid backlight or hand shadows

#📷 攝影機解析度建議為 640x480 / Recommended camera resolution: 640x480

#🔧 系統需求 / Requirements
Python >= 3.7

OpenCV (opencv-python)

TensorFlow (tensorflow)

MediaPipe (mediapipe)

NumPy

#📦 安裝套件 / Install dependencies:

bash
複製
編輯
pip install opencv-python mediapipe tensorflow numpy
#📦 模型檔案 / Model Files
請將以下模型放在 C:/hand/ 路徑，或依需要修改程式碼中的模型載入路徑。
Place the models under C:/hand/ or modify the path accordingly in the code:

best_model.h5：數字辨識模型 / Digit recognition model

symbol.h5：符號辨識模型（選用） / Symbol recognition model (optional)

augmented_model.h5：字母辨識模型 / Alphabet recognition model

#📄 授權 / License
本專案採用 MIT License 授權。
This project is licensed under the MIT License.
