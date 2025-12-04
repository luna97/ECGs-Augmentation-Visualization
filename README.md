# Streamlit ECG Augmentation Lab


This application allows for the visual inspection of 12-lead ECG augmentations. It supports building a pipeline of multiple augmentations and parametrizing them via a configuration file.

## Setup

1. **Install Dependencies**:
    ```bash
    pip install streamlit numpy scipy plotly wfdb pyyaml torch scikit-learn
    ```

2. **Data**:
   Ensure you have your WFDB compatible ECG files in a `data/` folder. The app currently looks for:
   - `data/heedb/...`
   - `data/code/...`
   *(You can modify `get_real_ecg` in `app.py` to point to your specific paths)*.

3. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## How to Add a New Augmentation

This application is data-driven. You do not need to modify `app.py` to add new augmentations or change slider ranges.

### Step 1: Add the Class
Open `augmentations.py` and add your Python class. 
*   It must accept a numpy array or torch tensor as input.
*   It must return a numpy array or torch tensor.

**Example:**
```python
# In augmentations.py
class ZeroOutFirstSample:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, x):
        x[0, :] = 0
        return x
```

### Step 2: Update Configuration
Open `config.yaml` and add an entry matching the **exact class name**. Define the parameters that match the class `__init__` arguments.

**Example:**
```yaml
# In config.yaml
augmentations:
  # ... existing augs ...
  
  ZeroOutFirstSample:
    description: "Sets the very first time step to zero."
    params:
      prob: { type: "float", default: 1.0, min: 0.0, max: 1.0, label: "Probability" }
```

### Special Parameters
The `app.py` automatically injects certain context variables if your class needs them. You do **not** need to add these to the YAML `params` list unless you want to override them manually.

- `fs` / `signal_fs`: The sampling frequency of the loaded ECG.
- `max_length`: Usually set to `fs * 10`.

If your class signature is:
```python
class MyFilter:
    def __init__(self, fs, cutoff): ...
```
Only put `cutoff` in `config.yaml`. The app will detect `fs` is missing from the config but present in the context and inject it automatically.