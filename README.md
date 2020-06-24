# PyTorch-OCR
This project detects and recognizes the text sequences from Chinese shop receipt/bill images. 

Note that this is based on the [chineseocr](https://github.com/chineseocr/chineseocr) and [darknet-ocr](https://github.com/wegamekinglc/darknet-ocr).

# Setup
- Ubuntu 18.04.4

- python 3.7.7 

- Install dependencies:
```
pip install -r requirements.txt
```

# Functions
- [x]  Text Detection  
- [x]  Sequence Recognition
- [x]  CPU/GPU Implementation 
- [ ]  Detection Model Training
- [ ]  Recognition Model Training

# Demo
For a simple example, please run:
```
python main.py
```

If the 'UnicodeEncodeError' occurs, please run:
```
PYTHONIOENCODING=utf-8 python main.py
```

# Web Application
- Run the code:
```
python app.py 8080
```

- Browse the web page: 
    
    [http://127.0.0.1:8080/text](http://127.0.0.1:8080/text)

- Click the '上传本地照片' button to upload a receipt/bill image. Click '识别' to recognize it.  

# Results



# References
- We convert the [chineseocr](https://github.com/chineseocr/chineseocr) that is implemented by Darknet framework to PyTorch.
- The web application setup refers to [darknet-ocr](https://github.com/wegamekinglc/darknet-ocr).


