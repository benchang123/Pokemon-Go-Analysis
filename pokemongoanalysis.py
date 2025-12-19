import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import os
from pathlib import Path
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class PokemonAnalyzer:
    """A comprehensive analyzer for Pokemon data comparison between games and Pokemon GO."""

    def __init__(self, data_path: str = None):
        if data_path is None:
            data_path = Path(__file__).parent
        self.data_path = Path(data_path)
        self.plots_dir = self.data_path / 'plots'
        self.poke_game = None
        self.poke_go = None
        self.poke_go_complete = None
        self.models = {}
        self.plot_counter = 0
        self.plots_dir.mkdir(exist_ok=True)

    def _save_plot(self, name: str) -> None:
        """Save current plot to file and close it."""
        self.plot_counter += 1
        filename = self.plots_dir / f"{self.plot_counter:02d}_{name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    def load_data(self) -> None:
        """Load Pokemon data from CSV files."""
        try:
            game_file = self.data_path / 'pokemoncomplete.csv'
            go_file = self.data_path / 'pogo.csv'

            self.poke_game = pd.read_csv(
                game_file, index_col=0, encoding='unicode_escape')
            self.poke_go = pd.read_csv(
                go_file, index_col=0, encoding='unicode_escape')

            print(f"Loaded {len(self.poke_game)} Pokemon from game data")
            print(f"Loaded {len(self.poke_go)} Pokemon from GO data")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def preprocess_data(self) -> None:
        """Clean and preprocess the Pokemon data."""
        if self.poke_game is None or self.poke_go is None:
            raise ValueError("Data must be loaded first. Call load_data()")

        self.poke_go = self.poke_go.reset_index()

        if 'Type 1' in self.poke_game.columns:
            self.poke_game.drop(columns=['Type 1', 'Type 2'], inplace=True)

        self.poke_game.rename(
            columns={
                'Attack': 'Attack_g',
                'Defense': 'Defense_g'
            },
            inplace=True)

        if 'Legendary' in self.poke_go.columns:
            self.poke_go.drop(columns=['Legendary'], inplace=True)

        legendary_indices = [647, 648, 649]
        mask = self.poke_game.index.isin(legendary_indices)
        if mask.any():
            self.poke_game.loc[mask, 'Legendary'] = True

        self.poke_go['Name'] = self.poke_go['Name'].str.lower().str.strip()
        self.poke_game['Name'] = self.poke_game['Name'].str.lower().str.strip()

        print("Data preprocessing completed")
        print("Pokemon GO missing values:", self.poke_go.isna().sum().sum())
        print("Pokemon Game missing values:", self.poke_game.isna().sum().sum())

    def merge_datasets(self) -> None:
        """Merge Pokemon GO and game datasets."""
        if self.poke_game is None or self.poke_go is None:
            raise ValueError("Data must be loaded and preprocessed first")

        self.poke_go_complete = self.poke_go.merge(
            self.poke_game, left_on='Name', right_on='Name', how='inner')

        if 'Generation_y' in self.poke_go_complete.columns:
            self.poke_go_complete.drop(columns=['Generation_y'], inplace=True)
        if 'Generation_x' in self.poke_go_complete.columns:
            self.poke_go_complete.rename(
                columns={'Generation_x': 'Generation'}, inplace=True)

        print(f"Successfully merged {len(self.poke_go_complete)} Pokemon")

    def create_visualizations(self) -> None:
        """Create comprehensive visualizations."""
        if self.poke_go_complete is None:
            raise ValueError("Data must be merged first")

        # Correlation analysis
        go_cols = ['MaxCP', 'Attack', 'Defense', 'Stamina']
        game_cols = [
            'MaxCP', 'HP', 'Attack_g', 'Defense_g', 'Sp. Atk', 'Sp. Def',
            'Speed'
        ]

        # Filter existing columns
        go_cols = [
            col for col in go_cols if col in self.poke_go_complete.columns
        ]
        game_cols = [
            col for col in game_cols if col in self.poke_go_complete.columns
        ]

        # Pokemon GO correlations
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.poke_go_complete[go_cols].corr(), annot=True, cmap='coolwarm')
        plt.title('Pokemon GO Stats Correlation')
        self._save_plot('go_stats_correlation')

        # Game stats correlations
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            self.poke_go_complete[game_cols].corr(),
            annot=True,
            cmap='coolwarm')
        plt.title('Pokemon Game Stats Correlation')
        self._save_plot('game_stats_correlation')

        # Distribution plots
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.poke_go_complete, x='MaxCP', kde=True)
        plt.title('Distribution of Max CP')
        self._save_plot('maxcp_distribution')

        # Type analysis
        plt.figure(figsize=(14, 8))
        ax = sns.boxplot(data=self.poke_go, x='Primary', y='MaxCP')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.title('Max CP by Primary Type')
        plt.tight_layout()
        self._save_plot('maxcp_by_type')

    @staticmethod
    def calculate_rmse(y_pred: np.ndarray, y_actual: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        return np.sqrt(mean_squared_error(y_actual, y_pred))

    def build_models(self) -> None:
        """Build prediction models."""
        if self.poke_go_complete is None:
            raise ValueError("Data must be merged first")

        # Game stats model
        feature_cols = [
            'HP', 'Attack_g', 'Defense_g', 'Sp. Atk', 'Sp. Def', 'Speed'
        ]
        available_cols = [
            col for col in feature_cols if col in self.poke_go_complete.columns
        ]

        if available_cols:
            X = self.poke_go_complete[available_cols].dropna()
            y = self.poke_go_complete.loc[X.index, 'MaxCP']

            model = linear_model.LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)

            rmse = self.calculate_rmse(predictions, y)
            print(f"Game Stats Model RMSE: {rmse:.2f}")
            print(f"Coefficients: {dict(zip(available_cols, model.coef_))}")

            self.models['game_stats'] = {
                'model': model,
                'features': available_cols
            }

        # Pokemon GO stats model
        go_features = ['Attack', 'Defense', 'Stamina']
        X_go = self.poke_go_complete[go_features].dropna()
        y_go = self.poke_go_complete.loc[X_go.index, 'MaxCP']

        model_go = linear_model.LinearRegression()
        model_go.fit(X_go, y_go)
        predictions_go = model_go.predict(X_go)

        rmse_go = self.calculate_rmse(predictions_go, y_go)
        print(f"Pokemon GO Stats Model RMSE: {rmse_go:.2f}")
        print(f"Coefficients: {dict(zip(go_features, model_go.coef_))}")

        self.models['go_stats'] = {'model': model_go, 'features': go_features}

    def analyze_strongest_pokemon(self) -> None:
        """Analyze strongest Pokemon using standardized stats."""
        if 'go_stats' not in self.models:
            print("Pokemon GO model not available")
            return

        # Standardize stats
        stats_cols = ['Attack', 'Defense', 'Stamina']
        stats_data = self.poke_go_complete[stats_cols].dropna()

        standardized = (stats_data - stats_data.mean()) / stats_data.std()
        model_coef = self.models['go_stats']['model'].coef_

        strength_scores = standardized @ model_coef

        # Create results dataframe
        results_df = self.poke_go_complete.loc[standardized.index].copy()
        results_df['Strength'] = strength_scores

        print("\n=== Strongest Pokemon (All) ===")
        top_all = results_df.nlargest(
            10, 'Strength')[['Name', 'Primary', 'Secondary', 'Strength']]
        print(top_all.to_string(index=False))

        print("\n=== Strongest Non-Legendary Pokemon ===")
        non_legendary = results_df[results_df['Legendary'] == False]
        top_non_legendary = non_legendary.nlargest(
            10, 'Strength')[['Name', 'Primary', 'Secondary', 'Strength']]
        print(top_non_legendary.to_string(index=False))

    def run_complete_analysis(self) -> None:
        """Run the complete Pokemon analysis pipeline."""
        print("Starting Pokemon Analysis...")

        self.load_data()
        self.preprocess_data()
        self.merge_datasets()
        self.create_visualizations()
        self.build_models()
        self.analyze_strongest_pokemon()

        print("\nAnalysis completed successfully!")


def main() -> None:
    """Main function to run Pokemon analysis."""
    # Use current directory or specify path
    analyzer = PokemonAnalyzer()

    try:
        analyzer.run_complete_analysis()
    except Exception as e:
        print(f"Analysis failed: {e}")


if __name__ == "__main__":
    main()
