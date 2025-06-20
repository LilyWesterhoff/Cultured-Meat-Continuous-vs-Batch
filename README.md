# Bioprocess Design Modeling Dashboard

This Streamlit application provides an interactive dashboard for bioprocess design modeling. It allows users to explore and compare batch and continuous bioprocesses using black box models.

## Features

- Interactive parameter adjustment for batch and continuous bioprocesses
- Real-time calculation of space-time yield factors and bioreactor purchase cost factors
- Visualization of results with comparative charts

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/bioprocess-design.git
   cd bioprocess-design
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Running the App

Start the Streamlit app:
```
streamlit run app.py
```

The app will be available at http://localhost:8501
Steps 2-3 can be automatically completed by running the setup shell script: zsh setup.sh 

## License

[MIT License](LICENSE)