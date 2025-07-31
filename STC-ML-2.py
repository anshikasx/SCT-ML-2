import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def load_customer_data(csv_file_path=None):
    """Load customer data from CSV file or use uploaded data"""
    if csv_file_path:
        try:
            df = pd.read_csv("/Users/anshikasinha/Downloads/Mall_Customers.csv")
            print(f"Successfully loaded {len(df)} records from {csv_file_path}")
            print(f"Columns in dataset: {list(df.columns)}")
            return df
        except FileNotFoundError:
            print(f"Error: File '{csv_file_path}' not found.")
            return None
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            return None
    else:
        # Try to load the uploaded Mall_Customers.csv using window.fs.readFile
        try:
            # This will work in the artifact environment
            import js
            file_content = js.window.fs.readFile('Mall_Customers.csv', {'encoding': 'utf8'})
            from io import StringIO
            df = pd.read_csv(StringIO(file_content))
            print(f"Successfully loaded {len(df)} records from uploaded file")
            print(f"Columns in dataset: {list(df.columns)}")
            return df
        except:
            # Fallback: create sample data matching your CSV structure
            print("Using sample data matching your CSV structure...")
            return generate_sample_data()

def generate_sample_data():
    """Generate sample data matching the Mall_Customers.csv structure"""
    np.random.seed(42)
    n_customers = 200
    
    customer_data = {
        'CustomerID': range(1, n_customers + 1),
        'Gender': np.random.choice(['Male', 'Female'], n_customers),
        'Age': np.random.randint(18, 71, n_customers),
        'Annual Income (k$)': np.random.randint(15, 140, n_customers),
        'Spending Score (1-100)': np.random.randint(1, 101, n_customers)
    }
    
    return pd.DataFrame(customer_data)

class CustomerSegmentation:
    def __init__(self, data, feature_columns=None):
        self.data = data.copy()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.kmeans_model = None
        self.optimal_k = None
        self.scaled_features = None
        self.feature_columns = feature_columns
        
    def preprocess_data(self):
        """Preprocess the mall customer data"""
        # Clean column names (remove special characters and spaces)
        self.data.columns = self.data.columns.str.replace(r'[^\w]', '_', regex=True)
        
        # Handle gender encoding if present
        if 'Gender' in self.data.columns:
            self.data['Gender_Encoded'] = self.label_encoder.fit_transform(self.data['Gender'])
        
        # Display basic info about the dataset
        print("Dataset Info:")
        print(f"Shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        print("\nFirst few rows:")
        print(self.data.head())
        print("\nBasic statistics:")
        print(self.data.describe())
        
    def detect_clustering_features(self):
        """Detect suitable features for clustering from mall customer data"""
        # For mall customer data, we'll use numerical features excluding CustomerID
        numerical_cols = []
        
        # Standard columns in mall customer dataset
        if 'Age' in self.data.columns:
            numerical_cols.append('Age')
        if 'Annual_Income__k__' in self.data.columns:  # After cleaning column name
            numerical_cols.append('Annual_Income__k__')
        elif 'Annual Income (k$)' in self.data.columns:
            numerical_cols.append('Annual Income (k$)')
        if 'Spending_Score__1_100_' in self.data.columns:  # After cleaning column name
            numerical_cols.append('Spending_Score__1_100_')
        elif 'Spending Score (1-100)' in self.data.columns:
            numerical_cols.append('Spending Score (1-100)')
        if 'Gender_Encoded' in self.data.columns:
            numerical_cols.append('Gender_Encoded')
            
        print(f"Selected features for clustering: {numerical_cols}")
        return numerical_cols
        
    def prepare_features(self, feature_columns=None):
        """Prepare and scale features for clustering"""
        if feature_columns is None:
            feature_columns = self.detect_clustering_features()
        
        # Validate that columns exist in the dataset
        missing_cols = [col for col in feature_columns if col not in self.data.columns]
        if missing_cols:
            print(f"Warning: The following columns are not in the dataset: {missing_cols}")
            feature_columns = [col for col in feature_columns if col in self.data.columns]
        
        if not feature_columns:
            raise ValueError("No valid features found for clustering")
        
        features = self.data[feature_columns]
        
        # Handle any missing values
        features = features.fillna(features.median())
        
        # Scale features
        self.scaled_features = self.scaler.fit_transform(features)
        self.feature_columns = feature_columns
        
        print(f"Features prepared for clustering: {feature_columns}")
        return self.scaled_features
    
    def find_optimal_k(self, max_k=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        if self.scaled_features is None:
            self.prepare_features()
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(self.data) // 2))  # Ensure k is reasonable
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_features)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_features, kmeans.labels_))
        
        # Find optimal k based on silhouette score
        best_silhouette_idx = np.argmax(silhouette_scores)
        self.optimal_k = k_range[best_silhouette_idx]
        
        return {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k': self.optimal_k
        }
    
    def perform_clustering(self, n_clusters=None):
        """Perform K-means clustering"""
        if self.scaled_features is None:
            self.prepare_features()
        
        if n_clusters is None:
            if self.optimal_k is None:
                self.find_optimal_k()
            n_clusters = self.optimal_k
        
        # Perform K-means clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(self.scaled_features)
        
        # Add cluster labels to original data
        self.data['Cluster'] = cluster_labels
        
        return cluster_labels
    
    def analyze_clusters(self):
        """Analyze and interpret clusters"""
        if 'Cluster' not in self.data.columns:
            self.perform_clustering()
        
        # Create analysis based on available columns
        analysis_columns = []
        for col in ['Age', 'Annual Income (k$)', 'Annual_Income__k__', 
                   'Spending Score (1-100)', 'Spending_Score__1_100_']:
            if col in self.data.columns:
                analysis_columns.append(col)
        
        if 'Gender' in self.data.columns:
            # Gender distribution by cluster
            gender_dist = pd.crosstab(self.data['Cluster'], self.data['Gender'])
            print("Gender distribution by cluster:")
            print(gender_dist)
            print()
        
        # Numerical analysis
        cluster_summary = self.data.groupby('Cluster')[analysis_columns].agg(['mean', 'std', 'min', 'max']).round(2)
        
        # Add cluster size
        cluster_sizes = self.data['Cluster'].value_counts().sort_index()
        
        return cluster_summary, cluster_sizes
    
    def interpret_clusters(self):
        """Provide business interpretation of clusters for mall customers"""
        cluster_summary, cluster_sizes = self.analyze_clusters()
        interpretations = {}
        
        # Get column names (handle cleaned names)
        income_col = 'Annual Income (k$)' if 'Annual Income (k$)' in self.data.columns else 'Annual_Income__k__'
        spending_col = 'Spending Score (1-100)' if 'Spending Score (1-100)' in self.data.columns else 'Spending_Score__1_100_'
        
        for cluster_id in range(len(cluster_summary)):
            if income_col in cluster_summary.columns.get_level_values(0):
                avg_income = cluster_summary.loc[cluster_id, (income_col, 'mean')]
                avg_spending = cluster_summary.loc[cluster_id, (spending_col, 'mean')]
                avg_age = cluster_summary.loc[cluster_id, ('Age', 'mean')]
            else:
                # Fallback if columns are different
                avg_income = 50  # Default values
                avg_spending = 50
                avg_age = 35
            
            # Classify based on income and spending patterns
            if avg_spending >= 70 and avg_income >= 70:
                segment = "High Value Customers"
                description = "High income, high spending. Target with premium products and exclusive offers."
                marketing_strategy = "Premium marketing, VIP services, luxury product recommendations"
            elif avg_spending >= 70 and avg_income < 70:
                segment = "Aspirational Shoppers"
                description = "High spending despite moderate income. Focus on trendy, aspirational products."
                marketing_strategy = "Trendy products, installment plans, social media marketing"
            elif avg_spending < 30 and avg_income >= 70:
                segment = "Careful Spenders"
                description = "High income but low spending. Conservative shoppers who need convincing."
                marketing_strategy = "Quality emphasis, detailed product information, trust-building"
            elif avg_spending < 30 and avg_income < 30:
                segment = "Budget Conscious"
                description = "Low income, low spending. Price-sensitive customers."
                marketing_strategy = "Discounts, promotions, value products, loyalty programs"
            elif 30 <= avg_spending < 70:
                segment = "Moderate Customers"
                description = "Balanced spending habits. Regular customers with steady purchasing power."
                marketing_strategy = "Balanced approach, seasonal promotions, product variety"
            else:
                segment = "Standard Customers"
                description = "Average customers with typical spending patterns."
                marketing_strategy = "General marketing campaigns, customer retention programs"
            
            interpretations[cluster_id] = {
                'segment_name': segment,
                'description': description,
                'marketing_strategy': marketing_strategy,
                'size': cluster_sizes[cluster_id],
                'avg_income': avg_income,
                'avg_spending': avg_spending,
                'avg_age': avg_age
            }
        
        return interpretations
    
    def visualize_clusters(self):
        """Create visualizations for cluster analysis"""
        if 'Cluster' not in self.data.columns:
            self.perform_clustering()
        
        # Handle column names
        income_col = 'Annual Income (k$)' if 'Annual Income (k$)' in self.data.columns else 'Annual_Income__k__'
        spending_col = 'Spending Score (1-100)' if 'Spending Score (1-100)' in self.data.columns else 'Spending_Score__1_100_'
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Income vs Spending Score (main clustering visualization)
        scatter1 = axes[0, 0].scatter(self.data[income_col], self.data[spending_col], 
                                     c=self.data['Cluster'], cmap='viridis', alpha=0.7, s=60)
        axes[0, 0].set_xlabel('Annual Income (k$)')
        axes[0, 0].set_ylabel('Spending Score (1-100)')
        axes[0, 0].set_title('Customer Segments: Income vs Spending Score')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0, 0])
        
        # 2. Age vs Spending Score
        scatter2 = axes[0, 1].scatter(self.data['Age'], self.data[spending_col], 
                                     c=self.data['Cluster'], cmap='viridis', alpha=0.7, s=60)
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Spending Score (1-100)')
        axes[0, 1].set_title('Customer Segments: Age vs Spending Score')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[0, 1])
        
        # 3. Cluster size distribution
        cluster_counts = self.data['Cluster'].value_counts().sort_index()
        bars = axes[1, 0].bar(range(len(cluster_counts)), cluster_counts.values, 
                             color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(cluster_counts)])
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Number of Customers')
        axes[1, 0].set_title('Cluster Size Distribution')
        axes[1, 0].set_xticks(range(len(cluster_counts)))
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{int(height)}', ha='center', va='bottom')
        
        # 4. Average income and spending by cluster
        cluster_means = self.data.groupby('Cluster')[[income_col, spending_col]].mean()
        x = np.arange(len(cluster_means))
        width = 0.35
        
        bars1 = axes[1, 1].bar(x - width/2, cluster_means[income_col], width, 
                              label='Avg Income (k$)', alpha=0.8)
        bars2 = axes[1, 1].bar(x + width/2, cluster_means[spending_col], width, 
                              label='Avg Spending Score', alpha=0.8)
        
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Average Value')
        axes[1, 1].set_title('Average Income and Spending Score by Cluster')
        axes[1, 1].set_xticks(x)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def generate_detailed_report(self):
        """Generate a comprehensive analysis report"""
        print("="*60)
        print("CUSTOMER SEGMENTATION ANALYSIS REPORT")
        print("="*60)
        
        # Basic dataset info
        print(f"\nDATASET OVERVIEW:")
        print(f"Total Customers: {len(self.data)}")
        print(f"Number of Clusters: {len(self.data['Cluster'].unique())}")
        
        # Cluster interpretations
        interpretations = self.interpret_clusters()
        
        print(f"\nCLUSTER ANALYSIS:")
        print("-" * 40)
        
        for cluster_id, info in interpretations.items():
            print(f"\nCluster {cluster_id}: {info['segment_name']}")
            print(f"Size: {info['size']} customers ({info['size']/len(self.data)*100:.1f}%)")
            print(f"Average Age: {info['avg_age']:.1f} years")
            print(f"Average Income: ${info['avg_income']:.1f}k")
            print(f"Average Spending Score: {info['avg_spending']:.1f}/100")
            print(f"Description: {info['description']}")
            print(f"Marketing Strategy: {info['marketing_strategy']}")
        
        print(f"\nRECOMMENDATIONS:")
        print("-" * 40)
        print("1. Focus marketing budget on High Value Customers and Aspirational Shoppers")
        print("2. Develop loyalty programs for Moderate Customers")
        print("3. Create targeted discount campaigns for Budget Conscious customers")
        print("4. Implement retention strategies for Careful Spenders")
        print("5. Use age-based marketing for different demographic segments")

# Main execution function
def main():
    """Main function to run the customer segmentation analysis"""
    print("=== MALL CUSTOMER SEGMENTATION ANALYSIS ===\n")
    
    # Load the data
    df = load_customer_data()
    
    if df is None:
        print("Could not load data. Exiting.")
        return
    
    # Initialize the segmentation model
    segmentation = CustomerSegmentation(df)
    
    # Preprocess the data
    segmentation.preprocess_data()
    
    # Find optimal number of clusters
    print("\n" + "="*50)
    print("FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("="*50)
    optimization_results = segmentation.find_optimal_k()
    print(f"Optimal number of clusters: {optimization_results['optimal_k']}")
    
    # Show silhouette scores for different k values
    print("\nSilhouette scores for different k values:")
    for k, score in zip(optimization_results['k_range'], optimization_results['silhouette_scores']):
        print(f"k={k}: {score:.3f}")
    
    # Perform clustering
    print(f"\n" + "="*50)
    print("PERFORMING CLUSTERING")
    print("="*50)
    clusters = segmentation.perform_clustering()
    print(f"Clustering completed with {segmentation.optimal_k} clusters")
    
    # Generate detailed report
    segmentation.generate_detailed_report()
    
    # Create visualizations
    print(f"\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    segmentation.visualize_clusters()
    
    return segmentation

# Run the analysis
if __name__ == "__main__":
    segmentation_model = main()