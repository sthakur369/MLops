# ## Imports



import mlflow
import kfp
import kfp.components as comp
import requests
import kfp.dsl as dsl
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import io
from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import pickle







# ## 1: Fetching and setting the data



def prepare_data():
    import mlflow
    import pandas as pd
    import requests
    import io
    
    print("---- Inside prepare_data component ----")
    url = 'https://drive.google.com/uc?id=13Ebq8aiS-khJGCU6qH8xCsCbHqaJuggT'
    df = pd.read_csv(url)
    df = df.dropna()
    df.to_csv(f'data/final_df.csv', index=False)
    print("\n ---- data csv is saved to PV location /data/final_df.csv ----")


# ## 2: Training and Testing data split



def train_test_split():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    print("---- Inside train_test_split component ----")
    df = pd.read_csv(f'data/final_df.csv')
    # Split the data into features and target
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify = y, random_state=47)
    
    np.save(f'data/X_train.npy', X_train)
    np.save(f'data/X_test.npy', X_test)
    np.save(f'data/y_train.npy', y_train)
    np.save(f'data/y_test.npy', y_test)
    
    print("\n---- X_train ----")
    print("\n")
    print(X_train)
    
    print("\n---- X_test ----")
    print("\n")
    print(X_test)
    
    print("\n---- y_train ----")
    print("\n")
    print(y_train)
    
    print("\n---- y_test ----")
    print("\n")
    print(y_test)







# ## 3: Defining the Hyperparameters



def define_hyperparameter_and_model():
    print("---- Inside define_hyperparameter_and_model component ----")
    
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    import pandas as pd
    import io
    from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
    from sklearn import metrics
    from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
    import pickle
    import numpy as np
    
    
    tree_param_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                   'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]}

    knn_param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                      'weights': ['uniform', 'distance']}

    svm_param_grid = {'C': [0.1, 1, 10, 100],
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

    # Define the classifiers to be trained
    tree_clf = DecisionTreeClassifier(random_state=42)
    knn_clf = KNeighborsClassifier()
    svm_clf = SVC(random_state=42)
    
    # Save classifiers to file
    with open(f'data/tree_clf.pkl', 'wb') as f:
        pickle.dump(tree_clf, f)    

    with open(f'data/knn_clf.pkl', 'wb') as f:
        pickle.dump(knn_clf, f)
        
    with open(f'data/svm_clf.pkl', 'wb') as f:
        pickle.dump(svm_clf, f)
        
        
        
        
    
    # Save parameters to file
    with open(f'data/tree_param_grid.pkl', 'wb') as f:
        pickle.dump(tree_param_grid, f)
        
    with open(f'data/knn_param_grid.pkl', 'wb') as f:
        pickle.dump(knn_param_grid, f)
        
    with open(f'data/svm_param_grid.pkl', 'wb') as f:
        pickle.dump(svm_param_grid, f)
        







# ## 4: Traning the models



def training():
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    import pandas as pd
    from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
    from sklearn import metrics
    from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
    import pickle
    
    print("---- Inside training_basic_classifier component ----")    
    
    # Load parameters from file
    with open(f'data/tree_param_grid.pkl', 'rb') as f:
        tree_param_grid = pickle.load(f)
        
    with open(f'data/knn_param_grid.pkl', 'rb') as f:
        knn_param_grid = pickle.load(f)
        
    with open(f'data/svm_param_grid.pkl', 'rb') as f:
        svm_param_grid = pickle.load(f)
    
    # Load classifier from file
    with open(f'data/tree_clf.pkl', 'rb') as f:
        tree_clf = pickle.load(f)
        
    with open(f'data/knn_clf.pkl', 'rb') as f:
        knn_clf = pickle.load(f)
        
        
    with open(f'data/svm_clf.pkl', 'rb') as f:
        svm_clf = pickle.load(f)
    
    # Load X_train and y_train from file
    X_train = np.load(f'data/X_train.npy',allow_pickle=True)
    y_train = np.load(f'data/y_train.npy',allow_pickle=True)
    

    # Perform grid search with cross-validation for each classifier
    tree_grid_search = GridSearchCV(tree_clf, param_grid=tree_param_grid, cv=5,error_score='raise')
    print('grid search success')
    tree_grid_search.fit(X_train, y_train)
    print('tree fit success')

    knn_grid_search = GridSearchCV(knn_clf, param_grid=knn_param_grid, cv=5,error_score='raise')
    knn_grid_search.fit(X_train, y_train)

    svm_grid_search = GridSearchCV(svm_clf, param_grid=svm_param_grid, cv=5,error_score='raise')
    svm_grid_search.fit(X_train, y_train)
    
    
    # Saving the trained model in file
    with open(f'data/model1.pkl', 'wb') as f:
        pickle.dump(tree_grid_search, f)
    
    with open(f'data/model2.pkl', 'wb') as f:
        pickle.dump(knn_grid_search, f)
        
    with open(f'data/model3.pkl', 'wb') as f:
        pickle.dump(svm_grid_search, f)
    
    print("\n logistic regression classifier is trained on iris data and saved to PV location /data/model.pkl ----")







# ## 5: Predicting the test



def predict_on_test_data():
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
    from sklearn import metrics
    from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score    
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    import pickle
    
    print("---- Inside predict_on_test_data component ----")
    
    # Loading the trained model from file
    with open(f'data/model1.pkl','rb') as f:
        tree_grid_search = pickle.load(f)
        
    with open(f'data/model2.pkl','rb') as f:
        knn_grid_search = pickle.load(f)
    
    with open(f'data/model2.pkl','rb') as f:
        svm_grid_search = pickle.load(f)
        
        
    
    X_test = np.load(f'data/X_test.npy',allow_pickle=True)
    
    # Predictions
    y_pred_tree_grid_search = tree_grid_search.predict(X_test)
    y_pred_knn_grid_search = knn_grid_search.predict(X_test)
    y_pred_svm_grid_search = svm_grid_search.predict(X_test)
    
    
    
    # Save predicted model to file
    with open(f'data/y_pred_tree_grid_search.pkl', 'wb') as f:
        pickle.dump(y_pred_tree_grid_search, f)
        
    with open(f'data/y_pred_knn_grid_search.pkl', 'wb') as f:
        pickle.dump(y_pred_knn_grid_search, f)
        
    with open(f'data/y_pred_svm_grid_search.pkl', 'wb') as f:
        pickle.dump(y_pred_svm_grid_search, f)
        
    
    print("\n---- Predicted classes ----")
    print("\n")
    print(y_pred_tree_grid_search)
    print(y_pred_knn_grid_search)
    print(y_pred_svm_grid_search)
    







# ## 6: Getting the metrices from the model


def get_metrics():
    import mlflow
    import pandas as pd
    import pickle
    import numpy as np
    from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
    from sklearn import metrics
    from sklearn import metrics
    from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
    import requests

    print("---- Inside get_metrics component ----")
    mlflow.set_tracking_uri("http://host.docker.internal:5000/")
    mlflow.set_experiment("Kubeflow-MLFlow intergration Model")
    mlflow.end_run()
    

    # Load the model from file   

    with open(f'data/model1.pkl','rb') as f:
        tree_grid_search = pickle.load(f)

    with open(f'data/model2.pkl','rb') as f:
        knn_grid_search = pickle.load(f)

    with open(f'data/model3.pkl','rb') as f:
        svm_grid_search = pickle.load(f)


    y_test = np.load(f'data/y_test.npy',allow_pickle=True)

    
    # Load the predicted model from file   
    with open(f'data/y_pred_tree_grid_search.pkl','rb') as f:
        y_pred_tree_grid_search = pickle.load(f)

    with open(f'data/y_pred_knn_grid_search.pkl','rb') as f:
        y_pred_knn_grid_search = pickle.load(f)

    with open(f'data/y_pred_svm_grid_search.pkl','rb') as f:
        y_pred_svm_grid_search = pickle.load(f)


    print("Decision Tree - all metrics:")
    tree_scores = cross_val_score(tree_grid_search.best_estimator_, y_test, y_pred_tree_grid_search, cv=5)
    tree_acc_mean = tree_scores.mean()
    tree_acc_std = tree_scores.std()
    tree_acc = accuracy_score(y_test, y_pred_tree_grid_search)
    print("Accuracy: {:.2f}".format(tree_acc))
    print("Best hyperparameters: ", tree_grid_search.best_params_)


    # Start first MLflow run

    with mlflow.start_run(run_name='Decision Tree matrices'):
    # Log the metrics in MLFlow(Decision_Tree)
        mlflow.log_param("model", "Decision Tree")
        mlflow.log_param("Decision_Tree_test_size", 0.2)
        mlflow.log_param("Decision_Tree_random_state", 42)

    #     mlflow.log_metric("Decision_Tree_tree_scores", tree_scores)
    #     mlflow.log_metric("Decision_Tree_tree_acc_mean", tree_acc_mean)
    #     mlflow.log_metric("Decision_Tree_tree_acc_std", tree_acc_std)
    #     mlflow.log_metric("Decision_Tree_r2", r2_score(y_test, y_pred_tree_grid_search))

        mlflow.log_metric("Decision_Tree_tree_acc", tree_acc)
        mlflow.log_params(tree_grid_search.best_params_)
        #mlflow.log_metrics(metrics.classification_report(y_test, tree_grid_search,output_dict = True))

    mlflow.end_run()
    # End the first run

    print(metrics.classification_report(y_test, y_pred_tree_grid_search,output_dict = True))

    
    print("\nKNN - all metrics:")
    knn_scores = cross_val_score(tree_grid_search.best_estimator_, y_test, y_pred_knn_grid_search, cv=5)
    knn_acc_mean = knn_scores.mean()
    knn_acc_std = knn_scores.std()
    KNN_acc = accuracy_score(y_test, y_pred_knn_grid_search)
    print("Accuracy: {:.2f}".format(KNN_acc))
    print("Best hyperparameters: ", knn_grid_search.best_params_)

    # Start second MLflow run
    with mlflow.start_run(run_name='KNN matrices'):
    # Log the metrics in MLFlow(KNN)
        mlflow.log_param("model", "KNN")
        mlflow.log_param("KNN_test_size", 0.2)

    #     mlflow.log_metric("KNN_tree_scores", knn_scores)
    #     mlflow.log_metric("KNN_tree_acc_mean", knn_acc_mean)
    #     mlflow.log_metric("KNN_tree_acc_std", knn_acc_std)
    #     mlflow.log_metric("KNN_r2", r2_score(y_test, y_pred_knn_grid_search))

        mlflow.log_metric("KNN_tree_acc", KNN_acc)
        mlflow.log_params(knn_grid_search.best_params_)
        #mlflow.log_metric("KNN_metrics.classification_report", metrics.classification_report(y_test, knn_grid_search), json=True)
    mlflow.end_run()
    print(metrics.classification_report(y_test, y_pred_knn_grid_search, output_dict = True))

    
    
    print("\nSVM - all metrics:")
    svm_scores = cross_val_score(tree_grid_search.best_estimator_, y_test, y_pred_svm_grid_search, cv=5)
    svm_acc_mean = svm_scores.mean()
    svm_acc_std = svm_scores.std()
    SVM_acc = accuracy_score(y_test, y_pred_svm_grid_search)
    print("Accuracy: {:.2f}".format(SVM_acc))
    print("Best hyperparameters: ", svm_grid_search.best_params_)


    with mlflow.start_run(run_name='SVM matrices'):
    # Start third MLflow run
    # Log the metrics in MLFlow(SVM)
        mlflow.log_param("model", "SVM")
        mlflow.log_param("SVM_test_size", 0.2)
        mlflow.log_param("SVM_random_state", 42)

    #     mlflow.log_metric("SVM_tree_scores", svm_scores)
    #     mlflow.log_metric("SVM_tree_acc_mean", svm_acc_mean)
    #     mlflow.log_metric("SVM_tree_acc_std", svm_acc_std)
    #     mlflow.log_metric("SVM_r2", r2_score(y_test, y_pred_svm_grid_search))

        mlflow.log_metric("SVM_tree_acc", SVM_acc)
        mlflow.log_params(svm_grid_search.best_params_)
        #mlflow.log_metric("SVM_metrics.classification_report", metrics.classification_report(y_test, y_pred_svm_grid_search), json=True)
    mlflow.end_run()
    print(metrics.classification_report(y_test, y_pred_svm_grid_search, output_dict = True))



    # Save accuracy to file
    with open(f'data/tree_acc.pkl', 'wb') as f:
        pickle.dump(tree_acc, f)



    with open(f'data/SVM_acc.pkl', 'wb') as f:
        pickle.dump(SVM_acc, f)


    with open(f'data/KNN_acc.pkl', 'wb') as f:
        pickle.dump(KNN_acc, f)



    print(metrics.classification_report(y_test, y_pred_tree_grid_search))
    print(metrics.classification_report(y_test, y_pred_knn_grid_search))
    print(metrics.classification_report(y_test, y_pred_svm_grid_search))

    mlflow.end_run()
    







# ## 7: Finding the best model



def best_metrics_model():
    import mlflow
    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.metrics import accuracy_score,precision_score,recall_score,log_loss
    from sklearn import metrics
    from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

    mlflow.set_tracking_uri("http://host.docker.internal:5000") # Replace <minikube-ip> with your Minikube IP address
    mlflow.set_experiment("Kubeflow-MLFlow intergration Model")
    mlflow.end_run()

        # Find the best model based on accuracy score

        # Load classifier from file
    with open(f'data/tree_clf.pkl', 'rb') as f:
        tree_clf = pickle.load(f)

    with open(f'data/knn_clf.pkl', 'rb') as f:
        knn_clf = pickle.load(f)


    with open(f'data/svm_clf.pkl', 'rb') as f:
        svm_clf = pickle.load(f)


    with open(f'data/model1.pkl','rb') as f:
        tree_grid_search = pickle.load(f)

    with open(f'data/model2.pkl','rb') as f:
        knn_grid_search = pickle.load(f)

    with open(f'data/model3.pkl','rb') as f:
        svm_grid_search = pickle.load(f)



    # Load accuracy from file
    with open(f'data/tree_acc.pkl','rb') as f:
        tree_acc = pickle.load(f)

    with open(f'data/SVM_acc.pkl','rb') as f:
        svm_acc = pickle.load(f)

    with open(f'data/KNN_acc.pkl','rb') as f:
        knn_acc = pickle.load(f)







    best_acc = max(tree_acc, knn_acc, svm_acc)
    best_model = None

    if best_acc == tree_acc:
        best_model = tree_clf
        best_params = tree_grid_search.best_params_
    elif best_acc == knn_acc:
        best_model = knn_clf
        best_params = knn_grid_search.best_params_
    else:
        best_model = svm_clf
        best_params = svm_grid_search.best_params_

    # Print out the best model and its corresponding hyperparameters
    print('best_accuracy: ', best_acc)
    print('best_model: ', best_model)
    print('best_params: ', best_params)


    # Start fourth MLflow run
    with mlflow.start_run(run_name='BEST MODEL and MATRICES'):
        mlflow.log_metric("best_accuracy", best_acc)
        mlflow.log_param("best_model", best_model)
        mlflow.log_params(best_params)
        mlflow.sklearn.log_model(best_model, "model")

    mlflow.end_run() 

    







# ## Kubeflow 7 Pipelines creation



create_step_prepare_data = kfp.components.create_component_from_func(
    func=prepare_data,
    base_image='python:3.9',
    packages_to_install=['pandas==1.4.2','numpy==1.21.5','requests==2.27.1','mlflow==2.3.1']
)


# In[10]:


create_step_train_test_split = kfp.components.create_component_from_func(
    func=train_test_split,
    base_image='python:3.9',
    packages_to_install=['pandas==1.4.2','numpy==1.21.5','scikit-learn==1.0.2']
)


# In[11]:


create_define_hyperparameter_and_model = kfp.components.create_component_from_func(
    func=define_hyperparameter_and_model,
    base_image='python:3.9',
    packages_to_install=['pandas==1.4.2','numpy==1.21.5','scikit-learn==1.0.2']
)


# In[12]:


create_step_training = kfp.components.create_component_from_func(
    func=training,
    base_image='python:3.9',
    packages_to_install=['pandas==1.4.2','numpy==1.21.5','scikit-learn==1.0.2']
)


# In[13]:


create_step_predict_on_test_data = kfp.components.create_component_from_func(
    func=predict_on_test_data,
    base_image='python:3.9',
    packages_to_install=['pandas==1.4.2','numpy==1.21.5','scikit-learn==1.0.2']
)


# In[14]:


create_step_get_metrics = kfp.components.create_component_from_func(
    func=get_metrics,
    base_image='python:3.9',
    packages_to_install=['pandas==1.4.2','numpy==1.21.5','scikit-learn==1.0.2','mlflow==2.3.1','requests==2.27.1']
)




create_step_best_metrics_model = kfp.components.create_component_from_func(
    func=best_metrics_model,
    base_image='python:3.9',
    packages_to_install=['pandas==1.4.2','numpy==1.21.5','scikit-learn==1.0.2','mlflow==2.3.1']
)







# ## Pipeline configs



# Define the pipeline
@dsl.pipeline(
   name='IRIS-classifier Kubeflow Pipeline',
   description='A sample pipeline that performs IRIS classifier task'
)
# Define parameters to be fed into pipeline
def iris_classifier_pipeline(data_path: str):
    vop = dsl.VolumeOp(
    name="t-vol",
    resource_name="t-vol", 
    size="1Gi", 
    modes=dsl.VOLUME_MODE_RWO)
    
  #  Setting the priority
    prepare_data_task = create_step_prepare_data().add_pvolumes({data_path: vop.volume})
    train_test_split = create_step_train_test_split().add_pvolumes({data_path: vop.volume}).after(prepare_data_task)
    define_hyperparameter_and_model = create_define_hyperparameter_and_model().add_pvolumes({data_path: vop.volume}).after(train_test_split)

    training = create_step_training().add_pvolumes({data_path: vop.volume}).after(define_hyperparameter_and_model)
    log_predicted_class = create_step_predict_on_test_data().add_pvolumes({data_path: vop.volume}).after(training)
    log_metrics_task = create_step_get_metrics().add_pvolumes({data_path: vop.volume}).after(log_predicted_class)
    log_best_metrics_task = create_step_best_metrics_model().add_pvolumes({data_path: vop.volume}).after(log_metrics_task)

    #setting caching config
    prepare_data_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    train_test_split.execution_options.caching_strategy.max_cache_staleness = "P0D"
    define_hyperparameter_and_model.execution_options.caching_strategy.max_cache_staleness = "P0D"
    training.execution_options.caching_strategy.max_cache_staleness = "P0D"
    log_predicted_class.execution_options.caching_strategy.max_cache_staleness = "P0D"
    log_metrics_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    log_best_metrics_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    







# ## Compiling and starting the pipeline



kfp.compiler.Compiler().compile(
    pipeline_func=iris_classifier_pipeline,
    package_path='IRIS-classifier Kubeflow Pipeline.yaml')

#Start
client = kfp.Client()









DATA_PATH = '/data'

import datetime
print(datetime.datetime.now().date())


pipeline_func = iris_classifier_pipeline
experiment_name = 'iris_classifier_exp' +"_"+ str(datetime.datetime.now().date())
run_name = pipeline_func.__name__ + ' run'
namespace = "kubeflow"

arguments = {"data_path":DATA_PATH}

kfp.compiler.Compiler().compile(pipeline_func,  
  '{}.zip'.format(experiment_name))

run_result = client.create_run_from_pipeline_func(pipeline_func, 
                                                  experiment_name=experiment_name, 
                                                  run_name=run_name, 
                                                  arguments=arguments)






