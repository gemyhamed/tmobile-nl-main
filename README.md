### Features

Task 1: Build a predictive model for product02.
Task 2 (Bonus): Push source code to a Git hosting platform.
Task 3 (Bonus): Create a Dockerfile to containerize the project.

### Prerequisites
Python 3.x
Required dependencies in requirements.txt


### How to Run
Clone the repo.
Install dependencies: pip install -r requirements.txt.
To train the model, use: python main.py train --data path_to_data
To predict using the model, use: python main.py predict --data path_to_data

### Docker Setup
Build the Docker image: docker build -t tmobile-nl .
Run the container: docker run -it tmobile-nl

### Files
main.py: Main training and prediction script.
requirements.txt: Dependencies.
playground.ipynb: Interactive notebook for model exploration.
Dockerfile: Container setup.
