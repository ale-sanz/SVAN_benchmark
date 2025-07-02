import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


'''bash script to create csv file from extracted insertion type, repeat class from vamos, and ME family when available:
awk -F'\t' '
BEGIN {
    OFS = "\t";
    print "ITYPE", "Class", "FAM_N";
}
$NF > 0 {
    # Extract ITYPE
    itype = ($5 ~ /ITYPE_N=/) ? gensub(/.*ITYPE_N=([^;]+).*/, "\\1", 1, $5) : "No_ITYPE";
    
    # Extract Class
    class = $9;

    # Extract FAM_N
    fam = ($5 ~ /FAM_N=/) ? gensub(/.*FAM_N=([^;]+).*/, "\\1", 1, $5) : "No_ME_Family";
    if (fam == "") fam = "No_ME_Family";

    print itype, class, fam;
}' intersect_wao_HGSVC_vamos.tsv > overlap_categories.tsv'''



outDir="/users/brodriguez/asanchez/scripts/VNTR/comparison"

def group_itype(itype):
##Combine all duplication insertion types into a single category, as well as solo and partnered insertions
    if any(prefix in itype for prefix in ['COMPLEX_DUP', 'DUP', 'DUP_INTERSPERSED', 'INV_DUP']):
        return 'DUPLICATION'
    elif itype.startswith(('solo', 'partnered')):
        return 'SOLO_PARTNERED'
    return itype

#Convert the extracted data from the intersect output into a dataframe
df = pd.read_csv('overlap_categories.tsv', sep='\t')

# Apply grouping function to the dataframe
df['Grouped_ITYPE'] = df['ITYPE'].apply(group_itype)

# Create count table
counts = df.groupby(['Grouped_ITYPE', 'Class']).size().unstack(fill_value=0)

# Sort by total count in descending order
counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]

##Create stacked boxed plot that shows the composition of intersections between vamos motifs and SVAN annotations based on insertion type.
plt.figure(figsize=(14, 8))
ax = plt.gca()
colors = {'STR': '#3498db', 'VNTR': '#e67e22'}
counts.plot(kind='bar', stacked=True, 
           color=[colors.get(x, '#95a5a6') for x in counts.columns],
           width=0.8, ax=ax)

total = counts.sum().sum()
for i, (index, row) in enumerate(counts.iterrows()):
    total_height = row.sum()
    percentage = (total_height / total * 100).round(2)
    
    ax.text(i, total_height * 1.02, f"{percentage}%",
           ha='center', va='bottom',
           color='black', fontsize=10,
           bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

plt.title('Vamos motifs Intersections with SVAN Insertion Categories', fontsize=16, pad=20)
plt.xlabel('Insertion Categories', fontsize=12)
plt.ylabel('Count', fontsize=12, labelpad=10)
plt.xticks(rotation=45, ha='right', fontsize=10)

plt.legend(title='Repeat Type', fontsize=10, title_fontsize=11, 
          bbox_to_anchor=(1.02, 1), loc='upper left')

plt.savefig(f'{outDir}/plots/repeats_per_itype_barplot.svg', bbox_inches='tight')

