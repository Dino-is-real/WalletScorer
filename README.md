# WalletScorer

**WalletScorer** is a Python tool for assigning credit scores (0-1000) to DeFi wallets based on their historical transaction behavior. It is designed for transparency, extensibility, and robust analysis of wallet reliability and risk.

## Features

- **Feature Engineering**: Extracts wallet-level features such as transaction frequency, asset diversity, volume, regularity, and action ratios (deposit, borrow, repay, etc.).
- **Synthetic Labeling**: Generates synthetic credit scores using a transparent, rule-based formula.
- **Machine Learning Model**: Trains a Random Forest Regressor on engineered features to predict wallet credit scores.
- **Score Tiers**: Classifies wallets into tiers (Excellent, Good, Fair, Poor, Very Poor) for easy interpretation.
- **Comprehensive Analysis**: Includes plotting and dashboard functions for score distribution, feature importances, and wallet behavior insights.

## How It Works

1. **Load Data**: Reads a JSON file of transaction records.
2. **Feature Engineering**: Aggregates transaction data per wallet, computing metrics like:
   - Total transactions
   - Unique assets interacted with
   - Time span of activity
   - Transaction frequency and regularity
   - Deposit/borrow/repay ratios
   - Volume and volatility
   - Borrow/repay balance
   - Asset diversity
3. **Synthetic Scoring**: Assigns a synthetic score to each wallet based on these features.
4. **Model Training**: Trains a Random Forest model to learn the mapping from features to scores.
5. **Prediction**: Outputs a credit score (0-1000) and tier for each wallet.
6. **Analysis & Visualization**: Generates plots and dashboards for further analysis.

## Usage

1. **Install dependencies**  
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn plotly
   ```

2. **Run the scorer**  
   Place your transaction JSON file in the working directory, then run:
   ```python
   from WalletScorer import AaveCreditScorer

   scorer = AaveCreditScorer()
   results = scorer.run_scoring('user_transactions.json', output_file='wallet_scores.json')
   ```

3. **Analyze results**  
   Use the included plotting and analysis functions to visualize score distributions and wallet behaviors.

## Output

- `wallet_scores.json`: List of wallet addresses with their credit score, tier, and key metrics.
- Plots: Score distribution, feature importances, correlation heatmaps, and more.
- Interactive dashboard: `wallet_analysis_dashboard.html` for in-depth exploration.

## Extensibility

- The scoring logic and feature engineering are modular and can be easily adapted for new data or alternative models.
- Swap out the synthetic labeler or model for your own logic or ML pipeline.

## Example

```python
scorer = AaveCreditScorer()
results = scorer.run_scoring('user_transactions.json', output_file='wallet_scores.json')
```

## License

MIT

---

**Note:**  
- For best results, use a large, diverse transaction dataset.
- The model is designed for interpretability and can be further tuned or replaced with more advanced ML techniques as needed.
