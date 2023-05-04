# MLops

## Intallations

### Docker Desktop

> Here we can see all the cotainers and images running

Follow this [Link](https://docs.docker.com/desktop/install/windows-install/) to install Docker Desktop

Now go to Settings -> Kubernetes and check 'Enable Kubernetes' and 'Show system containers (advanced)'

### Minikube

> Minikube is a tool that allows you to run a single-node Kubernetes cluster locally on your computer

Follow this [Link](https://minikube.sigs.k8s.io/docs/start/) to install Minikube

Now open Windows Powershell as Administrator and run these commands:

`$oldPath = [Environment]::GetEnvironmentVariable('Path', [EnvironmentVariableTarget]::Machine)
if ($oldPath.Split(';') -inotcontains 'C:\minikube'){ `
  [Environment]::SetEnvironmentVariable('Path', $('{0};C:\minikube' -f $oldPath), [EnvironmentVariableTarget]::Machine) `
}`

## Other Configs
```
pip install kfp

docker pull ghcr.io/mlflow/mlflow
```


## Installing MLflow and Kubeflow in one go
> I have given a file 'Kubeflow_and_MLFlow.yaml' which has full setup of MLflow and Kubeflow, you can use it but if you face any issues, you can install them seperatly >also on-by-one.
```
minikube start

kubectl create namespace mlflow

kubectl create -f Kubeflow_and_MLFlow.yaml
```

> OR you can install then one-by-one also



## Installing MLFlow in Minikube cluster

> Download the mlflow.yaml file and run these commands
```
minikube start

kubectl create namespace mlflow

kubectl create -f mlflow.yaml
```
## Installing Kubeflow in Minikube cluster
```
minikube start

set PIPELINE_VERSION=1.8.5

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"

kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"
```
It will 30-40 mins to complete, you can check the container getting created

`minikube kubectl -- get pods -A`

> if you see Error or CrashLoopBackOff dont worry it will get fixed automatically.

## Necessary addons
```
minikube addons enable metrics-server

minikube addons enable ingress
```
## Check Minikube Dashboard

Here you can see all the pods, serviecs, deployments running on minikube cluster.

`minikube dashboard`

## Check Kubeflow Dashboard
Here we will run our pipeline and we can see in the UI also
```
minikube tunnel
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```
now go to http://127.0.0.1:8080/

### Check MLFlow Dashboard
Here we will store our metrics and logs
```
minikube tunnel (if already started then no need)
kubectl port-forward -n mlflow service/mlflow-service 5000:80
```
now go to http://127.0.0.1:5000/

## To run the Pipeline

**Option 1:** 
Directly run the python file.

Python code will train and predict the model and also it will create pipelines in kubeflow and pass the metrics to MLFlow. So you can see Kubeflow and MLFlow are installed inside minikube and we can access their UI under same url but on different ports. But connection between MLFlow and Kubeflow is not possible that directly. For that we have to use this special IP to access MLFlow inside Kubeflow component: http://host.docker.internal:5000/.

**Option 2:** 
Download the '_IRIS-classifier Kubeflow Pipeline.yaml_' file and go to Pipeline section in Kubeflow dashboard and create new pipeline by uploading this yaml file. 

