# Random Forest Regressor using Scikit-Learn on Diamond dataset

About this repo:

- This is a random forest regressor model built using SciKit-Learn on the `diamond` dataset.
- The model has been trained already and the trained model artifacts are included.
- The repo contains the original dataset used for training and testing but you should generally avoid adding large data files in your repos. The data files are added here for convenience.
- There is a FastAPI service already set up for inference which you can start up by running the serve.py script inside the src folder. The service listens on port 80. The service offers two endpoints:

* `/ping`: This is a GET request. It should return a 200 status code when the service is healthy.
* `/predict`: This is a POST request. You should send a JSON formatted data with the feature names and values to be used as the sample for predict. Sample JSON data to send for inference looks as follows:

```
{
  "Id": "1782",
  "Carat Weight": 0.91,
  "Cut": "Very Good",
  "Color": "F",
  "Clarity": "SI1",
  "Polish": "VG",
  "Symmetry": "VG",
  "Report": "GIA"
}
```

Note that the request data should include the sample id (key "Id" in the exhibit above). The service will return the prediction - in this case, the predicted diamond value. Response data will look as follows:

```
{
  "data": {
    "Id": "1782",
    "Carat Weight": 0.91,
    "Cut": "Very Good",
    "Color": "F",
    "Clarity": "SI1",
    "Polish": "VG",
    "Symmetry": "VG",
    "Report": "GIA"
  },
  "prediction": 4213.5658
}
```

The FastAPI app also validates the data sent for prediction. The input fields must meet certain schema. Check the data_model.py file.

Your task for this exercise:

1. Create a Docker file for this app which should run the inference service when the container is run.
2. Build the image and run the container. Make sure you do the port mapping using the `-p` flag. Use the `-it` flag to see outputs in interactive mode.
3. Hit the two endpoints and make sure you get the expected responses. You can use Postman to make the HTTP requests. You can also directly go to http://127.0.0.1:80/docs or http://localhost:80/docs and see the list of APIs made available by the service. This is provided to us automatically by FastAPI using Swagger. Port 80 is default port used by most browsers, so you can skip it in the url.
4. Tag the image with the name you want and push it to your Docker Hub account.
