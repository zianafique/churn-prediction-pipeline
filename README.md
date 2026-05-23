# Churn Prediction ML Pipeline

## Setup

1. Create virtual environment:
```bash
   python -m venv venv
   source venv/bin/activate  

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Test AWS connection:
```bash
   python test_aws.py
```

## Project Structure
- `data/`: Raw and processed datasets
- `src/`: Main code (data processing, modeling, API)
- `notebooks/`: Jupyter notebooks for exploration
- `models/`: Trained model artifacts
- `tests/`: Unit tests

## Next Steps
- Download churn dataset (Step 3)
- Explore data (Step 4)