# Final Project: Handwriting gesture detection and recognition
### Malhar Mahant & Kruthika Gangaraju & Sriram Kodeeswaran
## Overview
An application that traces the tip of your finger to draw on a blank canvas and recognize text from the drawn image using Transformer based Optical Character Recognition.

## Important Links
1. [Github Repository](https://github.com/malhar212/finger-tracing-handwriting-recognition)
2. [Video Presentation](https://northeastern-my.sharepoint.com/:f:/r/personal/mahant_ma_northeastern_edu/Documents/CS%205330%20Final%20Project/video?csf=1&web=1&e=Tuyegw)
3. [Video Demonstration](https://northeastern-my.sharepoint.com/:f:/r/personal/mahant_ma_northeastern_edu/Documents/CS%205330%20Final%20Project/video?csf=1&web=1&e=Tuyegw)
4. [Data and other stuff](https://northeastern-my.sharepoint.com/:f:/r/personal/mahant_ma_northeastern_edu/Documents/CS%205330%20Final%20Project/video?csf=1&web=1&e=Tuyegw)

## Development Environment
Operating System: Windows 10 64 bit \
IDE: PyCharm  \
OpenCV version: 4.7.0

## Project Structure
Ensure the following files are in your directory.

```
│   main.py
│   tr_ocr.py
│   tr_ocr_experiments.ipynb
│   ReadME.md
│   Final_Project_Report.pdf
│   Project_Presentation.pptx
│   custom_trained_model
│       config.json
│       generation_config.json
│       pytorch_model.bin
```

## Model training and testing
Please open the python notebook `tr_ocr_experiments.ipynb` using a compatible software to view or run the model creation, training and testing code. <br>
Please note this requires additional files for the testing data available in data folder in the link mentioned above. <br>
Results in the notebook may vary slightly.


## How to run & use
1. Run the main.py file to use the application
2. Press 'd' to toggle drawing. Drawing will be enabled by default at the start.
3. Press 'c' to clear the canvas.
4. Press 'g' to clear the inference text.
5. When in drawing mode. Press 's' to save canvas image as training data. Enter the text (true label) of that image.
6. This image will be saved as a file in the `\data` directory and add an entry to `custom_data.csv` file
7. Press 'e' to switch to Evaluation Mode.
8. When in Evaluation mode. Press 's' to predict the text in the drawn canvas
9. Press 'q' to quit.
