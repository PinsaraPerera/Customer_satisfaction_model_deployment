# Customer_satisfaction_model_deployment

mlflow ui --backend-store-uri "file:C:\Users\USER\AppData\Roaming\zenml\local_stores\319fd890-7481-4266-8e1c-4c6c9e8c0e56\mlruns"

docker build -t zenml-app .
docker run -p 8237:8237 -v /d/Git\ projects/MLOps_Zenml/app:/app zenml-app
