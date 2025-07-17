!pip install -q scikit-learn pandas numpy
!pip install plotly --quiet

import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")



from google.colab import files
uploaded = files.upload()


filename = list(uploaded.keys())[0]

class AaveCreditScorer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def load_data(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def engineer_features(self, df):
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['amount_float'] = df['actionData'].apply(lambda x: float(x['amount']))
        df['asset_price'] = df['actionData'].apply(lambda x: float(x['assetPriceUSD']))
        df['usd_value'] = df['amount_float'] * df['asset_price']

        wallet_features = []

        for wallet in df['userWallet'].unique():
            wallet_data = df[df['userWallet'] == wallet].copy()
            total_txns = len(wallet_data)
            unique_assets = wallet_data['actionData'].apply(lambda x: x['assetSymbol']).nunique()
            time_span_days = (wallet_data['datetime'].max() - wallet_data['datetime'].min()).days + 1
            txn_frequency = total_txns / max(time_span_days, 1)

            action_counts = wallet_data['action'].value_counts()
            deposit_ratio = action_counts.get('deposit', 0) / total_txns
            borrow_ratio = action_counts.get('borrow', 0) / total_txns
            repay_ratio = action_counts.get('repay', 0) / total_txns

            total_volume = wallet_data['usd_value'].sum()
            avg_txn_size = wallet_data['usd_value'].mean()
            volume_std = wallet_data['usd_value'].std()

            time_diffs = wallet_data['datetime'].diff().dt.total_seconds().dropna()
            regularity_score = 1 - (time_diffs.std() / (time_diffs.mean() + 1e-8)) if len(time_diffs) > 1 else 0

            large_txn_ratio = (wallet_data['usd_value'] > wallet_data['usd_value'].quantile(0.95)).mean()

            if borrow_ratio > 0 and repay_ratio > 0:
                borrow_repay_balance = min(borrow_ratio, repay_ratio) / max(borrow_ratio, repay_ratio)
            else:
                borrow_repay_balance = 0

            wallet_features.append({
                'userWallet': wallet,
                'total_transactions': total_txns,
                'unique_assets': unique_assets,
                'time_span_days': time_span_days,
                'transaction_frequency': txn_frequency,
                'deposit_ratio': deposit_ratio,
                'borrow_ratio': borrow_ratio,
                'repay_ratio': repay_ratio,
                'total_volume_usd': total_volume,
                'avg_transaction_size': avg_txn_size,
                'volume_volatility': volume_std / (avg_txn_size + 1e-8),
                'regularity_score': regularity_score,
                'large_transaction_ratio': large_txn_ratio,
                'borrow_repay_balance': borrow_repay_balance,
                'asset_diversity': unique_assets / total_txns
            })

        return pd.DataFrame(wallet_features)

    def generate_synthetic_labels(self, features_df):
        scores = []
        for _, row in features_df.iterrows():
            score = 500
            score += min(row['total_transactions'] * 5, 100)
            score += min(row['unique_assets'] * 20, 100)
            score += min(row['time_span_days'] * 0.5, 100)
            score += row['borrow_repay_balance'] * 100
            score += (1 - row['regularity_score']) * 50
            score -= row['large_transaction_ratio'] * 100
            score -= max(0, row['volume_volatility'] - 1) * 50
            score -= max(0, row['transaction_frequency'] - 5) * 20
            score = max(0, min(1000, score))
            scores.append(score)
        return np.array(scores)

    def train_model(self, features_df):
        y = self.generate_synthetic_labels(features_df)
        feature_columns = [col for col in features_df.columns if col != 'userWallet']
        X = features_df[feature_columns].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)


        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\n📊 Feature Importances:")
        for i in indices:
            print(f"{feature_columns[i]}: {importances[i]:.4f}")


        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(feature_columns)), importances[indices], align='center')
        plt.xticks(range(len(feature_columns)), [feature_columns[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        return feature_columns

    def predict_scores(self, features_df, feature_columns):
        X = features_df[feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        scores = self.model.predict(X_scaled)
        return np.clip(scores, 0, 1000)

    def get_score_tier(self, score):
        if score >= 800:
            return 'Excellent'
        elif score >= 700:
            return 'Good'
        elif score >= 600:
            return 'Fair'
        elif score >= 500:
            return 'Poor'
        else:
            return 'Very Poor'

    def run_scoring(self, json_file, output_file='wallet_scores.json'):
        print("Loading transaction data...")
        df = self.load_data(json_file)

        print("Engineering features...")
        features_df = self.engineer_features(df)

        print("Training model...")
        feature_columns = self.train_model(features_df)

        print("Generating scores...")
        scores = self.predict_scores(features_df, feature_columns)

        results = []
        for i, (_, row) in enumerate(features_df.iterrows()):
            results.append({
                'wallet_address': row['userWallet'],
                'credit_score': int(scores[i]),
                'score_tier': self.get_score_tier(scores[i]),
                'key_metrics': {
                    'total_transactions': int(row['total_transactions']),
                    'unique_assets': int(row['unique_assets']),
                    'total_volume_usd': float(row['total_volume_usd']),
                    'time_span_days': int(row['time_span_days'])
                }
            })

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Scores saved to {output_file}")
        return results





scorer = AaveCreditScorer()
results = scorer.run_scoring(filename)


for result in results[:5]:
    print(f"Wallet: {result['wallet_address'][:20]}...")
    print(f"Score: {result['credit_score']} ({result['score_tier']})")
    print(f"Transactions: {result['key_metrics']['total_transactions']}")
    print("---")





def load_wallet_data(file_path):
    """Load wallet scores from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def create_score_ranges(df):
    """Create score ranges for analysis"""
    df['score_range'] = pd.cut(df['credit_score'],
        bins=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        labels=['0-100', '100-200', '200-300', '300-400', '400-500',
                '500-600', '600-700', '700-800', '800-900', '900-1000'],
        include_lowest=True)
    return df

def plot_score_distribution_pie(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    tier_counts = df['score_tier'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    ax1.pie(tier_counts.values, labels=tier_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 12})
    ax1.set_title('Credit Score Tier Distribution', fontsize=16, fontweight='bold')

    range_counts = df['score_range'].value_counts().sort_index()
    ax2.pie(range_counts.values, labels=range_counts.index, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Score Range Distribution', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('score_distribution_pie.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_score_histogram(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    n, bins, patches = ax.hist(df['credit_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')

    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < 500:
            patch.set_facecolor('#ff6b6b')
        elif bin_center < 600:
            patch.set_facecolor('#ffa500')
        elif bin_center < 700:
            patch.set_facecolor('#ffff00')
        elif bin_center < 800:
            patch.set_facecolor('#90ee90')
        else:
            patch.set_facecolor('#00ff00')

    ax.set_xlabel('Credit Score', fontsize=14)
    ax.set_ylabel('Number of Wallets', fontsize=14)
    ax.set_title('Distribution of Credit Scores', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)

    for line in [500, 600, 700, 800]:
        ax.axvline(x=line, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('score_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_score_vs_metrics_heatmap(df):
    metrics_df = pd.json_normalize(df['key_metrics'])
    correlation_data = pd.concat([df[['credit_score']], metrics_df], axis=1)
    corr_matrix = correlation_data.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                square=True, fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
    ax.set_title('Correlation Heatmap: Credit Score vs Key Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_metrics_by_tier(df):
    metrics_df = pd.json_normalize(df['key_metrics'])
    analysis_df = pd.concat([df[['score_tier']], metrics_df], axis=1)
    tier_order = ['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metrics = ['total_transactions', 'unique_assets', 'total_volume_usd', 'time_span_days']
    titles = ['Total Transactions', 'Unique Assets', 'Total Volume (USD)', 'Time Span (Days)']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        if metric == 'total_volume_usd':
            analysis_df[f'log_{metric}'] = np.log10(analysis_df[metric] + 1)
            sns.boxplot(data=analysis_df, x='score_tier', y=f'log_{metric}',
                        order=tier_order, ax=axes[i])
            axes[i].set_ylabel(f'Log10({title})')
        else:
            sns.boxplot(data=analysis_df, x='score_tier', y=metric,
                        order=tier_order, ax=axes[i])
            axes[i].set_ylabel(title)

        axes[i].set_xlabel('Score Tier')
        axes[i].set_title(f'{title} by Score Tier')
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('metrics_by_tier.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_score_range_analysis(df):
    df = create_score_ranges(df)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    range_counts = df['score_range'].value_counts().sort_index()
    axes[0, 0].bar(range(len(range_counts)), range_counts.values, color='lightblue', edgecolor='black')
    axes[0, 0].set_xticks(range(len(range_counts)))
    axes[0, 0].set_xticklabels(range_counts.index, rotation=45)
    axes[0, 0].set_ylabel('Number of Wallets')
    axes[0, 0].set_title('Wallet Count by Score Range')

    metrics_df = pd.json_normalize(df['key_metrics'])
    combined_df = pd.concat([df[['score_range']], metrics_df], axis=1)
    avg_metrics = combined_df.groupby('score_range').mean()

    axes[0, 1].plot(avg_metrics['total_transactions'], marker='o', label='Avg Transactions')
    axes[0, 1].plot(avg_metrics['unique_assets'], marker='s', label='Avg Assets')
    axes[0, 1].legend()
    axes[0, 1].set_title('Average Metrics by Score Range')

    combined_df['log_volume'] = np.log10(combined_df['total_volume_usd'] + 1)
    sns.violinplot(data=combined_df, x='score_range', y='log_volume', ax=axes[1, 0])
    axes[1, 0].set_title('Volume Distribution by Score Range')

    sns.boxplot(data=combined_df, x='score_range', y='time_span_days', ax=axes[1, 1])
    axes[1, 1].set_title('Activity Duration by Score Range')

    plt.tight_layout()
    plt.savefig('score_range_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_interactive_dashboard(df):
    metrics_df = pd.json_normalize(df['key_metrics'])
    combined_df = pd.concat([df, metrics_df], axis=1)

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=['Score Distribution', 'Score vs Volume', 'Score vs Transactions', 'Tier Averages'],
        specs=[[{"type": "bar"}, {"type": "scatter"}], [{"type": "scatter"}, {"type": "bar"}]]
    )

    fig.add_trace(go.Bar(x=df['score_tier'].value_counts().index,
                         y=df['score_tier'].value_counts().values,
                         marker_color='lightblue'), row=1, col=1)

    fig.add_trace(go.Scatter(x=combined_df['credit_score'],
                             y=np.log10(combined_df['total_volume_usd'] + 1),
                             mode='markers'), row=1, col=2)

    fig.add_trace(go.Scatter(x=combined_df['credit_score'],
                             y=combined_df['total_transactions'],
                             mode='markers'), row=2, col=1)

    avg_scores = df.groupby('score_tier')['credit_score'].mean()
    fig.add_trace(go.Bar(x=avg_scores.index, y=avg_scores.values, marker_color='orange'), row=2, col=2)

    fig.update_layout(title='Wallet Credit Score Analysis Dashboard', height=800)
    fig.write_html('wallet_analysis_dashboard.html')
    fig.show()

def generate_summary_statistics(df):
    print("="*60)
    print("WALLET CREDIT SCORE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total Wallets: {len(df)}")
    print(f"Average Score: {df['credit_score'].mean():.2f}")
    print(f"Median Score: {df['credit_score'].median():.2f}")
    print(f"Std Dev: {df['credit_score'].std():.2f}")
    print("\nTier Distribution:")
    print(df['score_tier'].value_counts())

def main():
    df = load_wallet_data('wallet_scores.json')
    generate_summary_statistics(df)

    print("\nGenerating plots...")
    df = create_score_ranges(df)
    plot_score_distribution_pie(df)
    plot_score_histogram(df)
    plot_score_vs_metrics_heatmap(df)
    plot_metrics_by_tier(df)
    plot_score_range_analysis(df)
    create_interactive_dashboard(df)

main()