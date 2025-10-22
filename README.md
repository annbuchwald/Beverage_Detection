# Application for detecting beverage containers!

## Running the Application

The application is deployed on Streamlit and available here: 
To run the app locally, you need to run the following commands:
```bash
# Install the requirements
pip install -r requirements.txt
# Start the app
streamlit run app/app.py
```
Now, you should be able to access the app on 

## Training the model

The application is based on the pre-trained [YOLOv8 model from Ultralytics](https://github.com/ultralytics/ultralytics) model, which was fine-tuned using the [
Beverage Containers Dataset](https://universe.roboflow.com/roboflow-universe-projects/beverage-containers-3atxb/dataset/3).

The code used for fine-tuning can be found in `/experiments/` directory along with metadata produced during the training inside `/experiments/train/`. After training the model, the weights inside `/experiments/train/weights/best.pt` were used for the application.

If you want to reproduce the fine-tuning, run the following steps:

```bash
# Install the requirements
pip install -r requirements.txt
# Run the training with your Roboflow API key
ROBOFLOW_API_KEY=<YOUR_API_KEY> python experiments/fine_tuning_script.py
```

Running training will produce run results in `/runs/detect/train`, where you can find your weights and training metadata.