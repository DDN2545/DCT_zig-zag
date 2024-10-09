from flask import Flask, render_template, request, send_from_directory, redirect
import os
import numpy as np
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/watermark'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'png'}

# สร้างโฟลเดอร์ถ้ายังไม่มี
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def dct2(block):
    return cv2.dct(np.float32(block))

def idct2(block):
    return cv2.idct(block)

def embed_watermark(original_image_path, watermark_image_path, output_path, watermark_strength=0.2):
    original = cv2.imread(original_image_path)
    watermark = cv2.imread(watermark_image_path)

    if original.shape[:2] != watermark.shape[:2]:
        watermark = cv2.resize(watermark, (original.shape[1], original.shape[0]))

    height, width, _ = original.shape
    block_size = 8
    watermarked = original.copy()

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            original_block = original[i:i + block_size, j:j + block_size]
            watermark_block = watermark[i:i + block_size, j:j + block_size]

            if original_block.shape[0] != block_size or original_block.shape[1] != block_size:
                continue

            dct_original_blocks = []
            for channel in range(3):
                dct_original = dct2(original_block[:, :, channel])
                dct_original[1][1] += watermark_strength * dct2(watermark_block[:, :, channel])[1][1]
                dct_original_blocks.append(dct_original)

            for channel in range(3):
                idct_block = idct2(dct_original_blocks[channel])
                idct_block = np.clip(idct_block, 0, 255)
                watermarked[i:i + block_size, j:j + block_size, channel] = idct_block

    cv2.imwrite(output_path, watermarked)

def remove_watermark(watermarked_image_path, watermark_strength=0.2):
    watermarked_image = cv2.imread(watermarked_image_path)
    height, width, _ = watermarked_image.shape
    block_size = 8
    original_image = np.zeros_like(watermarked_image)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            watermarked_block = watermarked_image[i:i + block_size, j:j + block_size]

            if watermarked_block.shape[0] != block_size or watermarked_block.shape[1] != block_size:
                continue

            for channel in range(3):
                dct_watermarked = dct2(watermarked_block[:, :, channel])
                dct_watermarked[1][1] -= watermark_strength * (dct_watermarked[1][1] - 0) * 0.0001  # ลบค่าความถี่ที่ถูกฝังลายน้ำออก
                idct_block = idct2(dct_watermarked)
                idct_block = np.clip(idct_block, 0, 255)
                original_image[i:i + block_size, j:j + block_size, channel] = idct_block

    return original_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'original_image' not in request.files or 'watermark_image' not in request.files:
            return redirect(request.url)

        original_file = request.files['original_image']
        watermark_file = request.files['watermark_image']
        strength = float(request.form['strength'])

        if original_file and allowed_file(original_file.filename) and watermark_file and allowed_file(watermark_file.filename):
            original_filename = secure_filename(original_file.filename)
            watermark_filename = secure_filename(watermark_file.filename)

            original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            watermark_path = os.path.join(app.config['UPLOAD_FOLDER'], watermark_filename)

            original_file.save(original_path)
            watermark_file.save(watermark_path)

            result_filename = 'watermarked_' + original_filename
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

            embed_watermark(original_path, watermark_path, result_path, watermark_strength=strength)

            return render_template('index.html', result_image=result_filename)

    return render_template('index.html', result_image=None)

@app.route('/remove_watermark', methods=['POST'])
def remove():
    if 'watermarked_image' not in request.files:
        return redirect(request.url)

    watermarked_file = request.files['watermarked_image']

    if watermarked_file and allowed_file(watermarked_file.filename):
        watermarked_filename = secure_filename(watermarked_file.filename)
        watermarked_path = os.path.join(app.config['UPLOAD_FOLDER'], watermarked_filename)

        watermarked_file.save(watermarked_path)

        # ลบลายน้ำออก
        original_image = remove_watermark(watermarked_path)

        # บันทึกภาพที่ไม่มีลายน้ำ
        original_filename = 'original_image.png'
        original_path = os.path.join(app.config['RESULT_FOLDER'], original_filename)
        cv2.imwrite(original_path, original_image)

        return render_template('index.html', original_image=original_filename)

    return redirect(request.url)

@app.route('/watermark/<filename>')
def send_watermarked_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
