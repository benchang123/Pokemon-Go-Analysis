# Pokemon GO Analysis

A data analysis project exploring the relationship between Pokemon statistics and Combat Power (CP) in Pokemon GO.

## Overview

This project analyzes Pokemon data to:
1. **Discover the CP Formula** - Identify which statistics correlate most strongly with a Pokemon's CP
2. **Predict Future CP** - Forecast CP values for upcoming Pokemon generations
3. **Rank Pokemon Strength** - Determine the strongest and weakest Pokemon in Pokemon GO

## Project Structure

```
Pokemon/
├── pokemongoanalysis.py    # Main analysis script (PokemonAnalyzer class)
├── Pokemon Go Writeup.ipynb # Jupyter notebook with detailed analysis
├── pogo.csv                 # Pokemon GO stats dataset
├── pokemoncomplete.csv      # Complete Pokemon game stats dataset
└── README.md
```

## Installation

### Requirements
- Python 3.7+
- Required packages:
  ```
  pandas
  numpy
  seaborn
  matplotlib
  scikit-learn
  ```

### Setup
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

## Usage

### Run the Complete Analysis
```python
from pokemongoanalysis import PokemonAnalyzer

analyzer = PokemonAnalyzer()
analyzer.run_complete_analysis()
```

### Run Individual Steps
```python
analyzer = PokemonAnalyzer()
analyzer.load_data()
analyzer.preprocess_data()
analyzer.merge_datasets()
analyzer.create_visualizations()
analyzer.build_models()
analyzer.analyze_strongest_pokemon()
```

### Command Line
```bash
python pokemongoanalysis.py
```

## Analysis Methods

### Data Processing
- Merges Pokemon GO stats with traditional game stats
- Handles missing values and normalizes Pokemon names
- Identifies legendary Pokemon for separate analysis

### Statistical Models
- **Linear Regression** on game stats (HP, Attack, Defense, Sp. Atk, Sp. Def, Speed)
- **Linear Regression** on Pokemon GO stats (Attack, Defense, Stamina)
- RMSE evaluation for model accuracy

### Visualizations
- Correlation heatmaps for both game and Pokemon GO stats
- CP distribution histograms
- Box plots of Max CP by Pokemon type

## Key Findings

The analysis reveals:
- Which base stats contribute most to CP calculation
- Correlation between traditional game stats and Pokemon GO performance
- Rankings of strongest Pokemon (legendary and non-legendary)

## Data Sources

- `pogo.csv` - Pokemon GO specific stats (MaxCP, Attack, Defense, Stamina, Types)
- `pokemoncomplete.csv` - Traditional Pokemon game stats across all generations

## License

This project is for educational and personal analysis purposes.
