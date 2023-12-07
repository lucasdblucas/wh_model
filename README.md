# wh_model
Algorithm used in the development of a method for estimating weight and height from multi-view images of an individual. The code supports both training and inference based on provided examples

## 📋 Tools

* Python 3.10
* Cuda 11.7
* Conda 23.3

### How to use

* First, download the Anaconda tool from the distributor's website.
```
https://www.anaconda.com/about-us
```

* Create a virtual environment from the file according to the project specifications.

```
conda env create -f environment.yml
```


* To download the project, clone it directly from git by executing the following command:

```


```
If not, go into the `cmad` folder
```
cd mad

```


| Atention: If you use a virtual environment, make sure it is activated!  <br/> If you have any questions, go to [link](https://www.treinaweb.com.br/blog/criando-ambientes-virtuais-para-projetos-python-com-o-virtualenv) |
| --- |

<h4>Installing and activating a virtual environment</h4>

* 1º - Installing

```
$ pip install virtualenv

```

* 2º - Creating an environment

```
$ virtualenv nome_da_virtualenv

```

* 3º - Activating the environment

Windows
```
$ nome_da_virtualenv/Scripts/Activate

```

Linux
```
$ source nome_da_virtualenv/bin/activate

```

* 4º - Install selenium

```
$ pip install selenium

```

* 5º - Run the python script

```
$ pip install selenium

```

The algorithm collects the following information about marks:

### Table of Columns
| Field Name | Description | Type |
| --- | --- | --- |
| Process Number | Unique identifier for the registration application. Used to access the image path if available. | Numeric |
| Mark Rejected | Name of the rejected mark. | Text |
| Trademark | Name of the registered trademark. | Text |
| Status | Trademark status (e.g., rejected, registered, waiting for analysis). | Text |
| Presentation | Type of mark presentation (e.g., Nominative, Figurative, Mixed). | Text |
| Nature | Mark nature (e.g., products, services). | Text |
| Nice Classification | International classification of goods and services for the marks' operational areas. | Text |
| Vienna Classification | System categorizing graphic elements in figurative, mixed, and three-dimensional marks. | Text |
| Application Date | Date when the mark was applied for. | Date |
| Complementary Text | Description of reasons for application denial. | Text |
| Magazine | Magazine publication number. | Text |


## 🤖 Consult our dataset via the link on zenodo

### [Zenodo](https://doi.org/10.5281/zenodo.10182880)

## 🤖 For more details, see our article on MDPI Data

### [MDPI Data](https://www.mdpi.com/journal/data)


## 👏 Contributing
 

CMAD code is an open-source project. If there is a bug, or other improvement you would like to report or request, we encourage you to contribute.

Please, feel free to contact us for any questions: [![Gmail Badge](https://img.shields.io/badge/-igor.bezerra@lsdi.ufma.br-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:igor.bezerra@lsdi.ufma.br)](mailto:igor.bezerra@lsdi.ufma.br)
