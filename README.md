# Heat Equation Gaussian Process Interpolation API
------

## 1. Table of Contents
- 2. [Installation](#installation)
- 3. [Usage](#usage)
- 4. [Perspectives](#perspectives)
- 5. [Authors and acknowledgment](#authors-and-acknowledgment)

## 2. Installation

After cloning the repo, build the image with:
```
$> docker build -t my-app:1.0 .
```

## 3. Usage

Start the container the following command:
```
$> docker run -p 8000:8000 my-app:1.0
```


### 3.1 Dashboard
Launch the dashboard with the following command:
```
$> streamlit run dashboard
```

## 4. Perspectives, ideas
[ ] Use Kubernetes to unify the dashboard and the API
[ ] Build Jenkins pipelines for test
[ ] Use of MPI to parallelize computations during DoE

## 5. Authors and acknowledgment
Pierre B. wrote this project based on lectures by Amina C., Jonas K. and Rodolphe L.