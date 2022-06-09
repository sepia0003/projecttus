# projecttus
Auto classification of TUS bulletin(offline & online) contents.

Please note that you have to download ChromeDriver from https://sites.google.com/chromium.org/driver/downloads and move it to "/usr/local/bin/chromedriver" (MAC)

The website which is powered by this project is http://sepia0003.pythonanywhere.com/    
Temporary id and pw: admin pw1122

1. Please run "preprocessing.py" to make vocabulary with collected and processed data.
2. Then run "model.py" to make a RNN model.
3. Then run "prediction.py" to classify collected data.
4. Then run "deploy.py" to make a Flask server.

Usage of preprocessing.py:

    (1) 
    input 'Y' to renew scrapped TUS online bulletin board.
    input 'N' to use previously scrapped one.
    input 'S' to skip TUS online bulletin board.

    (2)
    input 'Y' to renew saved TUS offline bulletin board.
    input 'N' to use previously saved one.
    input 'S' to skip TUS offline bulletin board.

    (3)
    input 'Y' to renew vocab with collected data.
    input 'N' to use previous vocab.

Please label all data before running model.py
