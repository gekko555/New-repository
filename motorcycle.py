import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
# バイクメーカーのクラス
classes = ["ZX10R", "KATANA", "CB1100R", "VMAX", "XL1200"]

# 画像サイズ
image_size = 150  # 画像サイズを整数値に変更
  # 学習に使用した画像サイズに合わせて設定

UPLOAD_FOLDER = "static"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

app = Flask(__name__)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model("./model.h5")  # 学習済みモデルのファイル名を指定

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # 受け取った画像を読み込み、予測モデルの入力サイズにリサイズ
            img = image.load_img(filepath, target_size=(224, 224))
            img = image.img_to_array(img)
            img = img / 255.0  # 画像の正規化
            data = np.array([img])
            
            # 以下は変更のない既存のコード
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "このバイクは " + classes[predicted] + " です"

            return render_template("index.html", answer=pred_answer)

    return render_template("index.html", answer="")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)