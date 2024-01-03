import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


filename = 'data/asu.tsv'
df = pd.read_table(filename, skiprows=54, sep=';', header=None, index_col=0,
                   names=['HIP', 'Vmag', 'Plx', 'B-V', 'SpType'],
                   skipfooter=1, engine='python')

# Remove rows with missing values
df_clean = df.applymap(lambda x: np.nan if isinstance(x, str) and x.isspace() else x).dropna()

# Convert columns to the appropriate data types
df_clean['Vmag'] = df_clean['Vmag'].astype(float)
df_clean['Plx'] = df_clean['Plx'].astype(float)
df_clean['B-V'] = df_clean['B-V'].astype(float)
df_clean['M_V'] = df_clean['Vmag'] + 5 * np.log10(df_clean['Plx'] / 100.)

# Function to map spectral type to class
def map_sptype_to_class(sptype):
    class_mapping = {
        'I': 'Supergiant',
        'II': 'Bright Giant',
        'III': 'Giant',
        'IV': 'Subgiant',
        'V': 'Main-sequence',
        'VI': 'Subdwarf',
        'VII': 'White Dwarf'
    }
    match = re.search(r'([IVXLCM]+)$', sptype)
    if match:
        roman_numeral = match.group(1)
        return class_mapping.get(roman_numeral, None)
    return None

# Apply the function to create the "Type" column
df_clean['Type'] = df_clean['SpType'].apply(map_sptype_to_class)

# Remove rows where the "Type" column is None
df_clean = df_clean.dropna(subset=['Type'])

df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
df_clean.dropna(axis=0, inplace=True)


# Select relevant columns for the features and target
features = df_clean[['M_V', 'Plx', 'B-V']]
target = df_clean['Type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=42)


# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Random Forest classifier
classifier = RandomForestClassifier(random_state=42)

classifier.fit(X_train_scaled, y_train)

y_pred = classifier.predict(X_test_scaled)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

clf_report = classification_report(y_test, y_pred)
print(clf_report)

def plot_hr_diagram(ax, data, title):
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(15, -15)
    ax.grid()
    ax.set_title(title)

    ax.title.set_fontsize(20)
    ax.set_xlabel('Color index B-V')
    ax.xaxis.label.set_fontsize(20)
    ax.set_ylabel('Absolute magnitude')
    ax.yaxis.label.set_fontsize(20)

    for value, color, label in zip(['White Dwarf', 'Subdwarf', 'Main-sequence', 'Subgiant', 'Giant', 'Bright Giant', 'Supergiant'], ['white', 'blue', 'black', 'grey', 'green', 'orange', 'yellow'],['VII: white dwarfs', 'VI: subdwarfs', 'V: main-sequence', 'IV: subgiants', 'III: giants', 'II: bright giants', 'I: supergiants']):
        if 'Type' in data.columns:
            b = data['Type'] == value
            x = data['B-V'][b]
            y = data['M_V'][b]
            ax.scatter(x, y, c=color, s=3, edgecolors='none', label=label)

    ax.tick_params(axis='both', labelsize=14)
    legend = ax.legend(scatterpoints=1, markerscale=6, shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    plt.show()

# Original Data Plot
fig, ax = plt.subplots(figsize=(8, 10))
plot_hr_diagram(ax, df_clean, 'H-R Diagram \n (Original Data)')

# Predicted Data Plot
fig, ax_test = plt.subplots(figsize=(8, 10))
# Assuming 'Type' is a predicted column name, adjust this based on your actual column name
plot_hr_diagram(ax_test, pd.DataFrame({'B-V': X_test['B-V'], 'M_V': X_test['M_V'], 'Type': y_test}),
                 'H-R Diagram \n (Test Data)')

# Predicted Data Plot
fig, ax_pred = plt.subplots(figsize=(8, 10))
# Assuming 'Type' is a predicted column name, adjust this based on your actual column name
plot_hr_diagram(ax_pred, pd.DataFrame({'B-V': X_test['B-V'], 'M_V': X_test['M_V'], 'Type': y_pred}),
                 'H-R Diagram \n (Predicted Data)')


