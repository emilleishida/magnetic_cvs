# magnetic_cataclysmic_variables
Fink science module to find magnetic cataclysmic variable stars.

## 🔧 Installation Guide

1. Clone the repository:
   ```bash
   git clone https://github.com/emilleishida/magnetic_cataclysmic_variables.git
   cd magnetic_cataclysmic_variables
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package locally:
   ```bash
   pip install .
   ```

## 🌌 Usage Example

```python
import mcvs

# Load your input DataFrame with object features
# Then evaluate candidates
results = mcvs.eval_candidates(df)
```

## 📧 Contact

If you have any question about some functionality of the package, do not hesitate to contact me at clems.mur@gmail.com.