# How to Run
On Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m streamlit run
```
On Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

# Model Training
- OCR Model Notebook: https://www.kaggle.com/code/technovore27/yolo-ocr 
- Number plate isolation model: https://colab.research.google.com/drive/1QDcIe1ICBNW5Kj4bdfclB8wEn1DR4wYH?usp=sharing
- Dataset used for Number plate isolation training: https://universe.roboflow.com/yolox-qcftu/indian-number-plate-keeo5/dataset/2
- Dataset used for OCR: https://universe.roboflow.com/car-plate-number-detection/alphanumeric-character-detection/dataset/1
