# 🍿 Movie Recommender System

This repository contains machine learning models of Movie Recommender System, designed to be deployed using ONNX and utilized in a Streamlit-based web application. The app provides an interactive interface for performing this task using neural network architectures. [Check here to see other ML tasks](https://github.com/verneylmavt/ml-model).

For more information about the training process, please check the `mov-recsys.ipynb` file in the `training` folder.

## 🎈 Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://verneylogyt-mov-recsys.streamlit.app/)

![Demo GIF](https://github.com/verneylmavt/st-mov-recsys/blob/main/assets/demo.gif)

<!-- <img src="https://github.com/verneylmavt/st-mov-recsys/blob/main/assets/demo.gif" alt="Demo GIF" width="750" height="750"> -->

If you encounter message `This app has gone to sleep due to inactivity`, click `Yes, get this app back up!` button to wake the app back up.

<!-- [https://verneylogyt.streamlit.app/](https://verneylogyt.streamlit.app/) -->

## ⚙️ Running Locally

If the demo page is not working, you can fork or clone this repository and run the application locally by following these steps:

<!-- ### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- pip (Python Package Installer)

### Installation Steps -->

1. Clone the repository:

   ```bash
   git clone https://github.com/verneylmavt/st-mov-recsys.git
   cd st-mov-recsys
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

Alternatively you can run `jupyter notebook demo.ipynb` for a minimal interface to quickly test the model (implemented w/ `ipywidgets`).

## ⚖️ Acknowledgement

I acknowledge the use of the **MovieLens 32M** dataset provided by **GroupLens Research** at the University of Minnesota. This dataset has been instrumental in conducting the research and developing this project.

- **Dataset Name**: MovieLens 32M
- **Source**: [https://grouplens.org/datasets/movielens/32m/](https://grouplens.org/datasets/movielens/32m/)
- **License**: The dataset may be used for research purposes under specific conditions; please refer to the [usage license](https://files.grouplens.org/datasets/movielens/ml-32m-README.html#usage-license) for details.
- **Description**: This dataset contains over 32 million ratings and 2 million tag applications applied to 87,585 movies by 200,948 users, collected from January 9, 1995, to October 12, 2023.

I deeply appreciate the efforts of GroupLens Research in making this dataset available.
